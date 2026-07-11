"""Retrieval evaluation and prediction export for news recommendation models.

Provides two evaluation modes:

- **retrieval**: Computes AUC, MRR, nDCG@k, HR@k, and Recall@k by scoring
  each user's impression list against their learned user vector.
- **dump_predictions**: Exports ranked prediction lists for offline analysis
  or competition submission.

For STAR models, user text is constructed on-the-fly by concatenating clicked
article titles. For LSNR and dense baselines, pre-computed news vectors are
looked up from a cache built at the start of evaluation.

Usage::

    uv run python src/evaluate.py --model_type lsnr --dataset mind --split test
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import gc
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent))

from data.dataset import NewsDataset, BehaviorsDataset
from utils import calculate_single_user_metric, log_metrics, set_seeds, get_device, load_model_and_checkpoint
from config import get_config


_STAR_NEWS_LOOKUP_CACHE = {}


def _build_star_news_lookup(news_path: str) -> dict:
    """Build a {news_id: {title, abstract}} lookup for STAR user text construction."""
    global _STAR_NEWS_LOOKUP_CACHE
    if news_path in _STAR_NEWS_LOOKUP_CACHE:
        return _STAR_NEWS_LOOKUP_CACHE[news_path]
    import pandas as pd
    try:
        df = pd.read_table(
            news_path,
            header=None,
            names=["id", "category", "subcategory", "title", "abstract",
                   "url", "title_entities", "abstract_entities"],
            usecols=["id", "title", "abstract"],
        )
    except Exception:
        df = pd.read_table(
            news_path,
            header=None,
            usecols=[0, 3, 4],
            names=["id", "title", "abstract"],
        )
    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    lookup = {str(row.id): {"title": row.title, "abstract": row.abstract}
            for row in df.itertuples()}
    _STAR_NEWS_LOOKUP_CACHE[news_path] = lookup
    return lookup


def _build_star_user_text(
    clicked_news_ids: list,
    news2data: dict,
    num_words_title: int = 20,
    num_words_abstract: int = 50,
    max_history: int = 10,
) -> str:
    """Reconstruct the STAR user text from click-history IDs (mirrors STARDataset logic)."""
    sorted_ids = list(reversed(clicked_news_ids))[:max_history]
    selected_texts = []
    for nid in sorted_ids:
        data = news2data.get(str(nid), {"title": "", "abstract": ""})
        title = " ".join(data["title"].split()[:num_words_title])
        abstract = " ".join(data["abstract"].split()[:num_words_abstract]) if data.get("abstract") else ""
        parts = [title] if title else []
        if abstract:
            parts.append(abstract)
        if parts:
            selected_texts.append(". ".join(parts))
    if not selected_texts:
        return "[HISTORY]"
    return "[HISTORY] " + ". ".join(selected_texts)

def build_news_vectors(model, config, device, split):
    ds_cfg = config['datasets'][config['global']['dataset']]
    mod_cfg = config['models'][config['global']['model_type']]
    train_cfg = config['training']
    
    news_dataset = NewsDataset(
        f"{ds_cfg['path']}/{split}/news.tsv", 
        ds_cfg['pretrained_model_name'], 
        config['global']['model_type'], 
        config['global']['dataset'], 
        ds_cfg['num_words_title'], 
        ds_cfg['num_words_abstract']
    )

    loader = DataLoader(news_dataset, batch_size=train_cfg['batch_size'] * 4, shuffle=False, num_workers=train_cfg['num_workers'])
    
    news2vector = {}
    news2category = {}
    model_type = config['global']['model_type']
    
    for batch in tqdm(loader, desc=f"Building News Vectors ({split})"):
        if config.get('global', {}).get('debug_mode', False) and len(news2vector) >= 200:
            print("\nDebug mode: stopping news vector construction early after 200 articles.")
            break
        news_ids = batch['id']
        if model_type in ['naml', 'lstur']:
            batch_data = {
                'input_ids': batch['input_ids'].squeeze(1).to(device),
                'attention_mask': batch['attention_mask'].squeeze(1).to(device),
                'category': batch['category'].to(device),
                'subcategory': batch['subcategory'].to(device)
            }
        elif model_type == 'star':
            batch_data = {
                'input_ids': batch['input_ids'].squeeze(1).to(device),
                'attention_mask': batch['attention_mask'].squeeze(1).to(device),
            }
        else:
            batch_data = batch
        
        vecs = model.get_news_vector(batch_data)
        
        if isinstance(vecs, tuple):
            vecs = (vecs[0].detach().cpu(), vecs[1].detach().cpu())
            for i, nid in enumerate(news_ids):
                nid_str = str(nid.item()) if isinstance(nid, torch.Tensor) else str(nid)
                news2vector[nid_str] = (vecs[0][i], vecs[1][i])
                news2category[nid_str] = int(batch['category'][i].item())
        else:
            vecs = vecs.detach().cpu()
            for i, nid in enumerate(news_ids):
                nid_str = str(nid.item()) if isinstance(nid, torch.Tensor) else str(nid)
                news2vector[nid_str] = vecs[i]
                news2category[nid_str] = int(batch['category'][i].item())
                
    if news2vector:
        first = next(iter(news2vector.values()))
        if isinstance(first, tuple):
            news2vector['PADDED_NEWS'] = (torch.zeros_like(first[0]), torch.zeros_like(first[1]))
        else:
            news2vector['PADDED_NEWS'] = torch.zeros_like(first)
        news2category['PADDED_NEWS'] = 0
            
    return news2vector, news2category

def _behaviors_collate_fn(batch):
    batch_out = {}
    max_clicked_len = max(len(item['clicked_news']) for item in batch)
    batch_out['clicked_news'] = [item['clicked_news'] + ['PADDED_NEWS'] * (max_clicked_len - len(item['clicked_news'])) for item in batch]
    batch_out['clicked_news_mask'] = torch.tensor([item['clicked_news_mask'] + [0] * (max_clicked_len - len(item['clicked_news_mask'])) for item in batch])
    batch_out['user'] = torch.tensor([item['user'] for item in batch], dtype=torch.long)
    batch_out['impression_id'] = [item.get('impression_id') for item in batch]
    batch_out['impressions'] = [item['impressions'] for item in batch]
    return batch_out

def eval_retrieval(model, config, device, news2vector, news2category, cold_threshold, split, batch_size, num_workers, max_samples=None):
    ds_cfg = config['datasets'][config['global']['dataset']]
    behaviors_file = f"{ds_cfg['path']}/{split}/behaviors_parsed.tsv"
    if not os.path.exists(behaviors_file): behaviors_file = f"{ds_cfg['path']}/{split}/behaviors.tsv"
            
    dataset = BehaviorsDataset(behaviors_file, cold_threshold=cold_threshold, 
                               num_clicked_news_a_user=int(ds_cfg.get('num_clicked_news_a_user', 30)),
                               min_history=int(config['global'].get('min_interactions', 0)))
            
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size * 4, num_workers=num_workers, collate_fn=_behaviors_collate_fn)

    metrics_sums = np.zeros(17)
    num_samples = 0
    nonzero, total = 0, 0
    model_type = config['global']['model_type']

    # --- STAR-specific setup: build news text lookup and tokenizer ---
    star_news2data = None
    star_tokenizer = None
    star_max_user_tokens = 256
    star_num_words_title = ds_cfg.get('num_words_title', 20)
    star_num_words_abstract = ds_cfg.get('num_words_abstract', 50)
    star_max_history = config.get('models', {}).get('star', {}).get('max_history_titles', 10)
    if model_type == 'star':
        news_path = f"{ds_cfg['path']}/{split}/news.tsv"
        star_news2data = _build_star_news_lookup(news_path)
        pretrained_name = (
            ds_cfg.get('star_pretrained_model_name') or
            config.get('models', {}).get('star', {}).get('pretrained_model_name') or
            ds_cfg.get('pretrained_model_name', 'naver/splade-cocondenser-ensembledistil')
        )
        star_tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        star_max_user_tokens = config.get('models', {}).get('star', {}).get('max_user_tokens', 256)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating ({split})"):
            if config.get('global', {}).get('debug_mode', False) and num_samples >= 20:
                print("\nDebug mode: stopping evaluation early after 20 samples.")
                break

            if max_samples is not None and num_samples >= max_samples:
                print(f"\nReached max_samples={max_samples}: stopping evaluation early.")
                break

            if model_type == 'star':
                # Build user text from click history on-the-fly and encode with SPLADE
                user_texts = [
                    _build_star_user_text(
                        c, star_news2data,
                        num_words_title=star_num_words_title,
                        num_words_abstract=star_num_words_abstract,
                        max_history=star_max_history,
                    )
                    for c in batch['clicked_news']
                ]
                tokenized = star_tokenizer(
                    user_texts,
                    padding=True,
                    truncation=True,
                    max_length=star_max_user_tokens,
                    return_tensors='pt',
                )
                user_vecs = model.get_user_vector({
                    'input_ids': tokenized['input_ids'].to(device),
                    'attention_mask': tokenized['attention_mask'].to(device),
                })
            elif model_type == 'lsnr':
                user_history = (
                    torch.stack([torch.stack([news2vector.get(nid, news2vector['PADDED_NEWS'])[1] for nid in c]) for c in batch['clicked_news']]).to(device),
                    torch.stack([torch.stack([news2vector.get(nid, news2vector['PADDED_NEWS'])[0] for nid in c]) for c in batch['clicked_news'] ]).to(device)
                )
                user_vecs = model.get_user_vector(user_history, mask=batch['clicked_news_mask'].to(device))
            else:
                user_history = torch.stack([torch.stack([news2vector.get(nid, news2vector['PADDED_NEWS']) for nid in c]) for c in batch['clicked_news']]).to(device)
                mask = batch['clicked_news_mask'].to(device)
                if model_type in ['lstur', 'npa']:
                    user_vecs = model.get_user_vector(batch['user'], user_history, mask=mask)
                else:
                    user_vecs = model.get_user_vector(user_history, mask=mask)

            uv_metric = user_vecs[0] if isinstance(user_vecs, tuple) else user_vecs
            nonzero += (uv_metric != 0).sum().item()
            total += uv_metric.numel()
            
            for i, user_id in enumerate(batch['user']):
                cands = [p.split('-') for p in batch['impressions'][i]]
                cand_ids = [p[0] for p in cands if p[0] != "PADDED_NEWS" and p[0] in news2vector]
                labels = [int(p[1]) for p in cands if p[0] != "PADDED_NEWS" and p[0] in news2vector]
                
                if not cand_ids or sum(labels) == 0 or sum(labels) == len(labels): continue
                
                cand_vecs = torch.stack([
                    news2vector.get(nid, news2vector['PADDED_NEWS'])[0] if model_type == 'lsnr'
                    else news2vector.get(nid, news2vector['PADDED_NEWS'])
                    for nid in cand_ids
                ]).to(device)
                scores = model.get_prediction(cand_vecs.unsqueeze(0), user_vecs[i].unsqueeze(0)).squeeze(0)
                
                m = calculate_single_user_metric((np.array(labels), scores.detach().cpu().numpy()))
                metrics_sums += m
                num_samples += 1

    avg_sparsity = 1.0 - (nonzero / total) if total > 0 else 0.0
    if num_samples == 0: return {"ret_auc": 0, "ret_mrr": 0, "ret_ndcg5": 0, "ret_ndcg10": 0, "avg_user_sparsity": avg_sparsity}
            
    res = metrics_sums / num_samples
    return {"ret_auc": res[0], "ret_mrr": res[1], "ret_ndcg5": res[2], "ret_ndcg10": res[3], "avg_user_sparsity": avg_sparsity}

def evaluate_model(model, config, device, split='val', prefix="", restore_train_mode=True):
    model.eval()
    train_cfg = config['training']
    cold_threshold = train_cfg['cold_threshold']
    
    news2vector, news2category = build_news_vectors(model, config, device, split)
    metrics = eval_retrieval(model, config, device, news2vector, news2category, cold_threshold, split, train_cfg['batch_size'], train_cfg['num_workers'])
    
    if prefix: log_metrics({f"{prefix}{k}": v for k, v in metrics.items()})
    if restore_train_mode: model.train()
    
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics

def dump_predictions(model, config, device, split, output_path):
    model.eval()
    train_cfg = config['training']
    news2vector, news2category = build_news_vectors(model, config, device, split)
    
    ds_cfg = config['datasets'][config['global']['dataset']]
    behaviors_file = f"{ds_cfg['path']}/{split}/behaviors_parsed.tsv"
    if not os.path.exists(behaviors_file): behaviors_file = f"{ds_cfg['path']}/{split}/behaviors.tsv"
    
    dataset = BehaviorsDataset(behaviors_file, cold_threshold=train_cfg['cold_threshold'], 
                               num_clicked_news_a_user=int(ds_cfg.get('num_clicked_news_a_user', 30)),
                               min_history=int(config['global'].get('min_interactions', 0)))
    
    loader = DataLoader(dataset, shuffle=False, batch_size=train_cfg['batch_size'] * 4, num_workers=train_cfg['num_workers'], collate_fn=_behaviors_collate_fn)
    model_type = config['global']['model_type']

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open('w') as f_out, torch.no_grad():
        f_out.write("[\n")
        first = True
        for batch in tqdm(loader, desc=f"Dumping Predictions ({split})"):
            if model_type == 'star': continue
            
            if model_type == 'lsnr':
                user_history = (
                    torch.stack([torch.stack([news2vector[nid][1] for nid in c]) for c in batch['clicked_news']]).to(device),
                    torch.stack([torch.stack([news2vector[nid][0] for nid in c]) for c in batch['clicked_news'] ]).to(device)
                )
            else:
                user_history = torch.stack([torch.stack([news2vector[nid] for nid in c]) for c in batch['clicked_news']]).to(device)
            
            if model_type in ['lstur', 'npa']:
                user_vecs = model.get_user_vector(batch['user'], user_history, mask=batch['clicked_news_mask'].to(device))
            else:
                user_vecs = model.get_user_vector(user_history, mask=batch['clicked_news_mask'].to(device))

            for i, user_id in enumerate(batch['user']):
                cands = [p.split('-')[0] for p in batch['impressions'][i] if p.split('-')[0] != 'PADDED_NEWS' and p.split('-')[0] in news2vector]
                if not cands: continue
                
                cand_vecs = torch.stack([news2vector[nid][0] if model_type == 'lsnr' else news2vector[nid] for nid in cands]).to(device)
                scores = model.get_prediction(cand_vecs.unsqueeze(0), user_vecs[i].unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                
                order = scores.argsort()[::-1]
                ranks = np.empty_like(order)
                ranks[order] = np.arange(1, len(scores) + 1)
                
                if not first: f_out.write(",\n")
                f_out.write(json.dumps({"impr_index": int(batch['impression_id'][i]), "pred_rank": ranks.tolist(), "pred_rel": scores.tolist()}))
                first = False
        f_out.write("\n]")

if __name__ == "__main__":
    config, args = get_config()
    set_seeds(config['global']['seed'])
    device = get_device(args.device)
    
    ckpt_dir = Path("./checkpoints") / args.experiment_name
    if not ckpt_dir.exists():
        candidates = [d for d in Path("./checkpoints").iterdir() if d.is_dir() and d.name.startswith(args.experiment_name)]
        ckpt_dir = candidates[0] if candidates else None
    
    model, _, _ = load_model_and_checkpoint(args.model_type, args.dataset, config, device, checkpoint_dir=str(ckpt_dir) if ckpt_dir else None)
    
    if args.eval_mode == "retrieval":
        metrics = evaluate_model(model, config, device, split=args.split, prefix=f"eval_{args.split}_")
        print("\nEvaluation Metrics:")
        for k, v in metrics.items(): print(f"{k}: {v:.4f}")
    else:
        dump_predictions(model, config, device, args.split, args.output_path)
        print(f"Predictions saved to {args.output_path}")
