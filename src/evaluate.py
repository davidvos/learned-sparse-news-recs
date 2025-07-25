"""
Evaluation script for news recommendation models.
Performs retrieval evaluation and builds news/user vectors for validation/testing.
"""

import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from multiprocessing import Pool
from utils import calculate_single_user_metric, latest_checkpoint, get_arg_parser, set_seeds, get_device
import argparse
import torch.nn.functional as F
from collections import defaultdict

from datasets import NewsDataset, BehaviorsDataset, UserDataset
from ebnerd_datasets import EbnerdNewsDataset, EbnerdBehaviorsDataset, EbnerdUserDataset

from model.LSNR import LSNR
from model.NAMLBERT import NAMLBERT
from model.LSTURBERT import LSTURBERT
from model.NRMSBERT import NRMSBERT

args = get_arg_parser().parse_args()

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

model_name = config[args.model]['model']
pretrained_model_name = config[args.dataset]['pretrained_model_name']

device = get_device()   # Select device (MPS, CUDA, or CPU)

# Model registry for easy model selection
MODEL_REGISTRY = {
    'nrms': NRMSBERT,
    'naml': NAMLBERT,
    'lstur': LSTURBERT,
    'lsnr': LSNR,
}

tokenizer = AutoTokenizer.from_pretrained(config[args.dataset]['pretrained_model_name'])

def eval_retrieval(model, news2vector, news2category, cold_threshold=0, args=None, filter_category=None):
    """
    Retrieval evaluation: For each user, rank all news articles and evaluate where the true clicked news ranks.
    Returns a tuple of averaged metrics.
    """
    with torch.no_grad():
        dataset_path = f"data/{args.dataset}"
        # Use ebnerd datasets if dataset is ebnerd
        if args.dataset in ['ebnerd', 'ebnerd_nyheder', 'ebnerd_sport']:
            behaviors_dataset = EbnerdBehaviorsDataset(f"{dataset_path}/val/behaviors_parsed_cold{cold_threshold}.tsv")
        else:
            behaviors_dataset = BehaviorsDataset(f"{dataset_path}/val/behaviors_parsed_cold{cold_threshold}.tsv")

        behaviors_loader = DataLoader(
            behaviors_dataset,
            shuffle=False,
            batch_size=args.batch_size * 16,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=behaviors_collate_fn
        )

        # Filter candidates by category if specified
        if filter_category is not None:
            candidate_ids = [nid for nid in news2vector if nid != "PADDED_NEWS" and news2category.get(nid, 0) == filter_category]
        else:
            candidate_ids = [nid for nid in news2vector if nid != "PADDED_NEWS"]
        candidate_vectors = torch.stack([news2vector[nid] for nid in candidate_ids]).to(device)
        news_to_idx = {nid: idx for idx, nid in enumerate(candidate_ids)}

        tasks = []
        for batch in tqdm(behaviors_loader, desc="Retrieval Evaluating"):
            # Build user history vectors for each user in the batch
            user_history_vectors = torch.stack([
                torch.stack([news2vector[nid] for nid in user_clicked_news])
                for user_clicked_news in batch['clicked_news']
            ]).to(device)

            # Get user vectors from the model
            if args.model == 'lstur':
                user_vectors = model.get_user_vector(batch['user'], user_history_vectors)
            else:
                user_vectors = model.get_user_vector(user_history_vectors)

            num_candidates = candidate_vectors.shape[0]
            vector_dim = candidate_vectors.shape[1]
            batch_size = user_vectors.shape[0]
            candidate_vectors_expanded = candidate_vectors.unsqueeze(0).expand(batch_size, num_candidates, vector_dim).to(user_vectors.device)
            # Add extra batch dimension to match model.get_prediction's unsqueeze(0)
            scores_flat = model.get_prediction(candidate_vectors_expanded, user_vectors)

            # For each user in the batch, get the true clicked news and its rank
            for i, (user_id, user_scores) in enumerate(zip(batch['user'], scores_flat)):
                labels = np.zeros(len(candidate_ids))
                for n in batch['impressions'][i]:
                    news_id, label = n.split('-')
                    if news_id in news_to_idx and int(label) == 1:
                        idx = news_to_idx[news_id]  # Use mapping instead of list.index()
                        labels[idx] = 1
                # Ensure user_scores is on CPU and numpy before passing to multiprocessing
                tasks.append((labels, user_scores.cpu().numpy()))

            # Free up memory
            del user_history_vectors, user_vectors, scores_flat
            torch.cuda.empty_cache()

    # Use multiprocessing to calculate metrics for each user
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(calculate_single_user_metric, tasks)
    results = np.array(results)
    aucs, mrrs, ndcg5s, ndcg10s, ndcg20s, ndcg50s, ndcg100s, hr5s, hr10s, hr20s, hr50s, hr100s, recall5s, recall10s, recall20s, recall50s, recall100s = results.T
    return (
        np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(ndcg10s), np.nanmean(ndcg20s), np.nanmean(ndcg50s), np.nanmean(ndcg100s),
        np.nanmean(hr5s), np.nanmean(hr10s), np.nanmean(hr20s), np.nanmean(hr50s), np.nanmean(hr100s),
        np.nanmean(recall5s), np.nanmean(recall10s), np.nanmean(recall20s), np.nanmean(recall50s), np.nanmean(recall100s)
    )


def build_news_vectors(model, config, args, cold_threshold=0):
    """
    Build news2vector and news2category dictionaries for all news articles in the validation set.
    Handles batching and device placement for efficiency.
    """
    dataset_path = f"data/{args.dataset}"
    # Use ebnerd datasets if dataset is ebnerd
    if args.dataset in ['ebnerd', 'ebnerd_nyheder', 'ebnerd_sport']:
        news_dataset = EbnerdNewsDataset(f"{dataset_path}/val/news.tsv", pretrained_model_name, args.model)
    else:
        news_dataset = NewsDataset(f"{dataset_path}/val/news.tsv", pretrained_model_name, args.model)

    news_dataloader = DataLoader(
        news_dataset,
        batch_size=args.batch_size * 16,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True
    )
    news2vector = {}
    news2category = {}

    # Process news in batches
    for batch in tqdm(news_dataloader, desc="Processing news"):
        news_ids = batch['id']
        news_text = batch['text']
        # Prepare batch data for model
        if args.model in ['naml', 'lstur']:
            batch_data = {
                'input_ids': batch['input_ids'].squeeze(1).to(device),
                'attention_mask': batch['attention_mask'].squeeze(1).to(device),
                'category': batch['category'].to(device),
                'subcategory': batch['subcategory'].to(device)
            }
        else:
            batch_data = batch
        # Get news vectors from the model
        news_vectors = model.get_news_vector(batch_data)
        news_vectors = news_vectors.detach().cpu()
        # Store vectors and categories
        if args.dataset in ['ebnerd', 'ebnerd_nyheder', 'ebnerd_sport']:
            for news_id, news_vector, category in zip(news_ids, news_vectors, batch['category']):
                news2vector[str(news_id.item())] = news_vector
                news2category[str(news_id.item())] = int(category.item())
        else:
            for news_id, news_vector, category in zip(news_ids, news_vectors, batch['category']):
                news2vector[str(news_id)] = news_vector
                news2category[str(news_id)] = int(category.item())
        del news_vectors, batch_data
        torch.cuda.empty_cache()
        
    # Add a zero vector for padding
    if news2vector:
        vector_dim = next(iter(news2vector.values())).shape[0]
        news2vector['PADDED_NEWS'] = torch.zeros(vector_dim)
        news2category['PADDED_NEWS'] = 0
    print(f"Processed {len(news2vector)} news vectors")
    return news2vector, news2category


def behaviors_collate_fn(batch):
    """
    Collate function for behaviors DataLoader. Pads clicked_news and clicked_news_mask to the max length in the batch.
    """
    batch_out = {}
    # Find max length for clicked_news in this batch
    max_clicked_len = max(len(item['clicked_news']) for item in batch)
    # Left pad clicked_news and clicked_news_mask to max_clicked_len
    batch_out['clicked_news'] = [
        ['PADDED_NEWS'] * (max_clicked_len - len(item['clicked_news'])) + item['clicked_news']
        for item in batch
    ]
    batch_out['clicked_news_mask'] = torch.tensor(
        [[0] * (max_clicked_len - len(item['clicked_news_mask'])) + item['clicked_news_mask'] for item in batch]
    )
    batch_out['clicked_news_length'] = torch.tensor([item['clicked_news_length'] for item in batch])
    batch_out['user'] = torch.tensor([item['user'] for item in batch], dtype=torch.long)
    batch_out['impressions'] = [item['impressions'] for item in batch]
    return batch_out


if __name__ == "__main__":

    # Set random seeds for reproducibility
    set_seeds()
    # Validate model selection
    model_name = args.model
    dataset = args.dataset
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {args.model}. Available models: {list(MODEL_REGISTRY.keys())}")

    # Instantiate model
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(config[args.model], config[args.dataset], device)

    # Load latest checkpoint
    checkpoint_path = latest_checkpoint(f'./checkpoints/{model_name}_{dataset}_cold{args.cold_threshold}_lr{args.learning_rate}/')

    if checkpoint_path is None:
        print('No checkpoint file found!')
    else:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    # Build news and user vectors once for both evaluations
    news2vector, news2category = build_news_vectors(model, config, args, cold_threshold=args.cold_threshold)

    # Set filter_category based on dataset. This is used for the retrieval evaluation in out-of-distribution settings.
    if args.dataset == 'mind_sports':
        filter_category = 156
    elif args.dataset == 'mind_news':
        filter_category = 254
    elif args.dataset == 'ebnerd_nyheder':
        filter_category = 1
    elif args.dataset == 'ebnerd_sport':
        filter_category = 10
    else:
        filter_category = None

    # Evaluate for multiple cold thresholds
    for cold_threshold_eval in [0, 1, 2, 3, 4, 5]:
        ret_auc, ret_mrr, ret_ndcg5, ret_ndcg10, ret_ndcg20, ret_ndcg50, ret_ndcg100, ret_hr5, ret_hr10, ret_hr20, ret_hr50, ret_hr100, ret_recall5, ret_recall10, ret_recall20, ret_recall50, ret_recall100 = eval_retrieval(
            model, news2vector, news2category, cold_threshold=cold_threshold_eval, args=args, filter_category=filter_category)
        print(f"Cold threshold: {cold_threshold_eval}")
        print({
            "ret_auc": ret_auc,
            "ret_mrr": ret_mrr,
            "ret_ndcg5": ret_ndcg5,
            "ret_ndcg10": ret_ndcg10,
            "ret_ndcg20": ret_ndcg20,
            "ret_ndcg50": ret_ndcg50,
            "ret_ndcg100": ret_ndcg100,
            "ret_hr5": ret_hr5,
            "ret_hr10": ret_hr10,
            "ret_hr20": ret_hr20,
            "ret_hr50": ret_hr50,
            "ret_hr100": ret_hr100,
            "ret_recall5": ret_recall5,
            "ret_recall10": ret_recall10,
            "ret_recall20": ret_recall20,
            "ret_recall50": ret_recall50,
            "ret_recall100": ret_recall100
        })

