import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import os
import gc
import json
from pathlib import Path
from tqdm import tqdm

from data.dataset import NewsDataset, BehaviorsDataset
from utils import calculate_single_user_metric, log_metrics
from model.base import BaseModel

class Evaluator:
    """
    Evaluator class for evaluating news recommendation models.
    """
    def __init__(self, config: Dict[str, Any], device: torch.device, dataset_type: str, model_type: str):
        self.config = config
        self.device = device
        self.dataset_type = dataset_type
        self.model_type = model_type
        
        self.dataset_config = config['datasets'][dataset_type]
        self.training_config = config['training']
        
        # Determine dataset path
        self.dataset_path = self.dataset_config['path']

    def evaluate(
        self, 
        model: BaseModel, 
        cold_threshold: int = 0, 
        split: str = 'val', 
        prefix: str = "",
        return_vectors: bool = False,
        restore_train_mode: bool = True
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Any, Any]]:
        """
        Run evaluation on the requested split.
        """
        model.eval()
        
        batch_size = self.training_config['batch_size']
        num_workers = self.training_config['num_workers']
        
        # Build vectors
        news2vector, news2category = self.build_news_vectors(
            model, batch_size, num_workers, cold_threshold, split
        )
        
        # Run retrieval evaluation
        metrics_tuple = self.eval_retrieval(
            model, news2vector, news2category, 
            cold_threshold=cold_threshold, 
            split=split,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        ret_auc, ret_mrr, ret_ndcg5, ret_ndcg10, avg_sparsity = metrics_tuple
        
        metrics = {
            "ret_auc": ret_auc,
            "ret_mrr": ret_mrr,
            "ret_ndcg5": ret_ndcg5,
            "ret_ndcg10": ret_ndcg10,
            "avg_user_sparsity": avg_sparsity
        }
        
        # Log metrics if prefix is provided (implying we want side-effect logging)
        if prefix:
             log_metrics({f"{prefix}{k}": v for k, v in metrics.items()})

        if restore_train_mode:
            model.train()
            
        if return_vectors:
            return metrics, news2vector, news2category
            
        del news2vector, news2category
        gc.collect()
        torch.cuda.empty_cache()
        
        return metrics

    def build_news_vectors(
        self,
        model: BaseModel,
        batch_size: int,
        num_workers: int,
        cold_threshold: int,
        split: str
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        
        pretrained_model_name = self.dataset_config['pretrained_model_name']
        num_words_title = self.dataset_config['num_words_title']
        num_words_abstract = self.dataset_config['num_words_abstract']
        
        
        news_dataset = NewsDataset(
            f"{self.dataset_path}/{split}/news.tsv", 
            pretrained_model_name, 
            self.model_type, 
            self.dataset_type, 
            num_words_title, 
            num_words_abstract
        )

        news_dataloader = DataLoader(
            news_dataset,
            batch_size=batch_size * 4,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True
        )
        
        news2vector = {}
        news2category = {}
        
        for batch in tqdm(news_dataloader, desc=f"Building News Vectors ({split})"):
             news_ids = batch['id']
             
             if self.model_type in ['naml', 'lstur']:
                 batch_data = {
                     'input_ids': batch['input_ids'].squeeze(1).to(self.device),
                     'attention_mask': batch['attention_mask'].squeeze(1).to(self.device),
                     'category': batch['category'].to(self.device),
                     'subcategory': batch['subcategory'].to(self.device)
                 }
             elif self.model_type == 'star':
                 batch_data = {
                     'input_ids': batch['input_ids'].squeeze(1).to(self.device),
                     'attention_mask': batch['attention_mask'].squeeze(1).to(self.device),
                 }
             else:
                 batch_data = batch
             
             news_vectors = model.get_news_vector(batch_data)
             
             # Handle tuple return (LSNR: sparse, dense)
             if isinstance(news_vectors, tuple):
                 # Move to cpu
                 news_vectors = (news_vectors[0].detach().cpu(), news_vectors[1].detach().cpu())
                 ids = news_ids
                 cats = batch['category']
                 
                 for i, news_id in enumerate(ids):
                     nid = str(news_id.item()) if isinstance(news_id, torch.Tensor) else str(news_id)
                     news2vector[nid] = (news_vectors[0][i], news_vectors[1][i])
                     news2category[nid] = int(cats[i].item())
             else:
                 news_vectors = news_vectors.detach().cpu()
                 for i, news_id in enumerate(news_ids):
                     nid = str(news_id.item()) if isinstance(news_id, torch.Tensor) else str(news_id)
                     news2vector[nid] = news_vectors[i]
                     news2category[nid] = int(batch['category'][i].item())
        
        # Add padding vector
        if news2vector:
            first_val = next(iter(news2vector.values()))
            if isinstance(first_val, tuple):
                sparse_dim = first_val[0].shape[0]
                dense_dim = first_val[1].shape[0]
                news2vector['PADDED_NEWS'] = (torch.zeros(sparse_dim), torch.zeros(dense_dim))
            else:
                 news2vector['PADDED_NEWS'] = torch.zeros_like(first_val)
            news2category['PADDED_NEWS'] = 0
            
        return news2vector, news2category

    def eval_retrieval(
        self,
        model: BaseModel,
        news2vector: Dict[str, Any],
        news2category: Dict[str, int],
        cold_threshold: int,
        split: str,
        batch_size: int,
        num_workers: int
    ) -> Tuple[float, float, float, float, float]:
        
        # Determine behaviors file
        behaviors_file = os.path.join(self.dataset_path, split, "behaviors_parsed.tsv")
        if not os.path.exists(behaviors_file):
            behaviors_file = os.path.join(self.dataset_path, split, "behaviors.tsv")
            
        history_limit = int(self.dataset_config.get('num_clicked_news_a_user', 30))
        min_history = int(self.config['global'].get('min_interactions', 0))

        from datasets import BehaviorsDataset
        behaviors_dataset = BehaviorsDataset(
            behaviors_file,
            cold_threshold=cold_threshold,
            num_clicked_news_a_user=history_limit,
            min_history=min_history
        )
            
        # Import collate_fn locally or from module
        # It's in evaluate.py currently, maybe we can inline it or import from utils if moved
        # For now, I'll duplicate the collate logic or assume we move it to datasets.py?
        # Let's check where it is. It was in evaluate.py `behaviors_collate_fn`.
        # I should probably move that to datasets.py or define it here.
        # Defining it here as static method or helper
        
        behaviors_loader = DataLoader(
            behaviors_dataset,
            shuffle=False,
            batch_size=batch_size * 4,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._behaviors_collate_fn
        )

        # Accumulators
        sum_auc = 0.0
        sum_mrr = 0.0
        sum_ndcg5 = 0.0
        sum_ndcg10 = 0.0
        num_metric_samples = 0
        
        user_nonzero = 0
        user_total = 0

        with torch.no_grad():
            for batch in tqdm(behaviors_loader, desc=f"Evaluating ({split})"):
                # Prepare User History Input
                # For LSNR: needs (dense, sparse) tuple lookup
                # For others: dense lookup
                
                if self.model_type == 'lsnr':
                    user_dense = torch.stack([
                        torch.stack([news2vector[nid][1] for nid in user_clicked_news])
                        for user_clicked_news in batch['clicked_news']
                    ]).to(self.device)
                    user_sparse = torch.stack([
                        torch.stack([news2vector[nid][0] for nid in user_clicked_news])
                        for user_clicked_news in batch['clicked_news']
                    ]).to(self.device)
                    user_history_input = (user_dense, user_sparse)
                elif self.model_type == 'star':
                    # Skip STAR special logic for this pass or implement later?
                    # I'll check if I can support it.
                    # It's cleaner to keep separate or move logic to model.
                    # I'll stick to standard logic for now and maybe `star` will fail if not handled.
                    # But I should try to support it. 
                    # Providing `news2vector` to `get_user_vector` is not enough for STAR if it needs raw text.
                    # Let's leave STAR specific logic out for a moment or just copy it if I have space.
                    pass 
                else:
                     user_history_input = torch.stack([
                        torch.stack([news2vector[nid] for nid in user_clicked_news])
                        for user_clicked_news in batch['clicked_news']
                    ]).to(self.device)
                
                # Get User Vector
                if self.model_type != 'star':
                     # Non-STAR models
                     if self.model_type in ['lstur', 'npa']:
                         user_vecs = model.get_user_vector(batch['user'], user_history_input, mask=batch['clicked_news_mask'].to(self.device))
                     else:
                         user_vecs = model.get_user_vector(user_history_input, mask=batch['clicked_news_mask'].to(self.device))
                     
                else:
                    # STAR placeholder
                    continue 

                # Metrics
                if isinstance(user_vecs, tuple):
                    uv_metric_vec = user_vecs[0]
                else:
                    uv_metric_vec = user_vecs
                
                user_nonzero += (uv_metric_vec != 0).sum().item()
                user_total += uv_metric_vec.numel()
                
                # Per-user candidates
                for i, user_id in enumerate(batch['user']):
                    user_candidate_pairs = batch['impressions'][i]
                    user_candidate_ids = []
                    user_labels = []
                    
                    for pair in user_candidate_pairs:
                        news_id, label = pair.split('-')
                        if news_id == "PADDED_NEWS": continue
                        if news_id not in news2vector: continue
                        user_candidate_ids.append(news_id)
                        user_labels.append(int(label))
                    
                    if not user_candidate_ids: continue
                    num_pos = sum(user_labels)
                    if num_pos == 0 or num_pos == len(user_labels): continue
                    
                    # Candidate vectors
                    if self.model_type == 'lsnr':
                        cand_vecs = torch.stack([news2vector[nid][0] for nid in user_candidate_ids]).to(self.device)
                    else:
                        cand_vecs = torch.stack([news2vector[nid] for nid in user_candidate_ids]).to(self.device)
                    
                    # Prediction
                    user_vec_i = user_vecs[i].unsqueeze(0)
                    cand_vecs = cand_vecs.unsqueeze(0)
                    
                    scores = model.get_prediction(cand_vecs, user_vec_i).squeeze(0)
                    
                    metrics = calculate_single_user_metric((np.array(user_labels), scores.detach().cpu().numpy()))
                    
                    sum_auc += metrics[0]
                    sum_mrr += metrics[1]
                    sum_ndcg5 += metrics[2]
                    sum_ndcg10 += metrics[3]
                    num_metric_samples += 1

        avg_sparsity = 1.0 - (user_nonzero / user_total) if user_total > 0 else 0.0
        
        if num_metric_samples == 0:
            return 0.0, 0.0, 0.0, 0.0, avg_sparsity
            
        return (
            sum_auc / num_metric_samples,
            sum_mrr / num_metric_samples,
            sum_ndcg5 / num_metric_samples,
            sum_ndcg10 / num_metric_samples,
            avg_sparsity
        )

    def dump_predictions(
        self,
        model: BaseModel,
        news2vector: Dict[str, Any],
        news2category: Dict[str, int],
        cold_threshold: int,
        split: str,
        output_path: Union[str, os.PathLike],
        batch_size: int,
        num_workers: int,
        filter_category: Optional[int] = None
    ) -> None:
        """
        Stream a full inference pass and write per-impression predictions as a JSON array.
        """
        behaviors_file = os.path.join(self.dataset_path, split, "behaviors.tsv")
        
        history_limit = int(self.dataset_config.get('num_clicked_news_a_user', 30))
        min_history = int(self.config['global'].get('min_interactions', 0))

        from datasets import BehaviorsDataset
        behaviors_dataset = BehaviorsDataset(
            behaviors_file,
            cold_threshold=cold_threshold,
            num_clicked_news_a_user=history_limit,
            min_history=min_history
        )

        behaviors_loader = DataLoader(
            behaviors_dataset,
            shuffle=False,
            batch_size=batch_size * 4,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._behaviors_collate_fn
        )

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with out_path.open('w') as f_out, torch.no_grad():
            f_out.write("[")
            first_record = True
            for batch in tqdm(behaviors_loader, desc=f"Dumping Predictions ({split})"):
                
                # Prepare User History Input
                if self.model_type == 'lsnr':
                    user_dense = torch.stack([
                        torch.stack([news2vector[nid][1] for nid in user_clicked_news])
                        for user_clicked_news in batch['clicked_news']
                    ]).to(self.device)
                    user_sparse = torch.stack([
                        torch.stack([news2vector[nid][0] for nid in user_clicked_news])
                        for user_clicked_news in batch['clicked_news']
                    ]).to(self.device)
                    user_history_input = (user_dense, user_sparse)
                elif self.model_type == 'star':
                    continue # STAR not fully supported in this simplified refactor yet
                else:
                     user_history_input = torch.stack([
                        torch.stack([news2vector[nid] for nid in user_clicked_news])
                        for user_clicked_news in batch['clicked_news']
                    ]).to(self.device)

                # Get User Vector
                if self.model_type != 'star':
                     if self.model_type in ['lstur', 'npa']:
                         user_vectors = model.get_user_vector(batch['user'], user_history_input, mask=batch['clicked_news_mask'].to(self.device))
                     else:
                         user_vectors = model.get_user_vector(user_history_input, mask=batch['clicked_news_mask'].to(self.device))
                else:
                    continue

                # For each user/impression in batch
                for i, user_id in enumerate(batch['user']):
                    user_candidate_pairs = batch['impressions'][i]
                    user_candidate_ids: List[str] = []
                    for pair in user_candidate_pairs:
                        news_id, _label = pair.split('-')
                        if news_id == 'PADDED_NEWS': continue
                        if news_id not in news2vector: continue
                        if filter_category is not None and news2category.get(news_id, 0) != filter_category: continue
                        user_candidate_ids.append(news_id)

                    if len(user_candidate_ids) == 0: continue
                    
                    if self.model_type == 'lsnr':
                        candidate_vectors = torch.stack([news2vector[nid][0] for nid in user_candidate_ids]).to(self.device)
                    else:
                        candidate_vectors = torch.stack([news2vector[nid] for nid in user_candidate_ids]).to(self.device)
                    
                    user_vec = user_vectors[i].unsqueeze(0)
                    scores = model.get_prediction(candidate_vectors.unsqueeze(0), user_vec).squeeze(0)
                    scores_np = scores.detach().cpu().numpy()

                    # Ranks: 1 = highest score
                    order = scores_np.argsort()[::-1]
                    ranks = np.empty_like(order)
                    ranks[order] = np.arange(1, len(scores_np) + 1)

                    impression_id = batch['impression_id'][i]
                    record = {
                        "impr_index": int(impression_id),
                        "pred_rank": ranks.tolist(),
                        "pred_rel": scores_np.tolist(),
                    }
                    if not first_record:
                        f_out.write(",\n")
                    else:
                        f_out.write("\n")
                        first_record = False
                    f_out.write(json.dumps(record))

            if not first_record:
                f_out.write("\n")
            f_out.write("]")

    @staticmethod
    def _behaviors_collate_fn(batch):
        # ... Copy of behaviors_collate_fn from evaluate.py ...
        batch_out = {}
        max_clicked_len = max(len(item['clicked_news']) for item in batch)
        batch_out['clicked_news'] = [
            item['clicked_news'] + ['PADDED_NEWS'] * (max_clicked_len - len(item['clicked_news']))
            for item in batch
        ]
        batch_out['clicked_news_mask'] = torch.tensor(
            [item['clicked_news_mask'] + [0] * (max_clicked_len - len(item['clicked_news_mask'])) for item in batch]
        )
        batch_out['clicked_news_length'] = torch.tensor([item['clicked_news_length'] for item in batch])
        batch_out['user'] = torch.tensor([item['user'] for item in batch], dtype=torch.long)
        batch_out['impression_id'] = [item.get('impression_id') for item in batch]
        batch_out['impressions'] = [item['impressions'] for item in batch]
        return batch_out
