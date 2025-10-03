#!/usr/bin/env python3
"""
Inspect LSNR news representations on MIND.

Loads an LSNR checkpoint per experiment and prints, for 10 sampled news:
- Title and abstract
- Top-10 dimensions of the LSNR news vector and their tokenizer tokens
"""

import os
import sys
import random
from typing import List, Tuple

import string
import numpy as np
import pandas as pd
import torch

STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
    "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
    'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't",
    'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't",
    'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
    "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll",
    "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's",
    'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't",
    'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that',
    "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these',
    'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't",
    'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom',
    'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've",
    'your', 'yours', 'yourself', 'yourselves', "'d", "'ll", "'m", "'re", "'s", "'t", "'ve"
}

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import load_experiment_config
from utils import latest_checkpoint, set_seeds, get_device
from evaluate import MODEL_REGISTRY


def load_model_and_config(experiment_name: str, dataset: str) -> Tuple[torch.nn.Module, torch.device, dict]:
    """Instantiate LSNR model and load latest checkpoint for the given experiment/dataset."""
    config = load_experiment_config(experiment_name, dataset)
    set_seeds(config['global']['seed'])
    device = get_device(config['global']['device'])

    model_type = config['experiment']['model']
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_type}' not found in registry. Expected 'lsnr'.")
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(config['models'][model_type], config['datasets'][dataset], device)

    # Resolve checkpoint dir using training lr and cold threshold for naming convention
    lr = config['training']['learning_rate']
    # Force cold0 to match available checkpoints
    cold_tr = 0
    ckpt_name = f"{experiment_name}_{model_type}_{dataset}_cold{cold_tr}_lr{lr}"
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', ckpt_name)
    ckpt_path = latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")

    # PyTorch 2.6 compatibility for weights_only default
    try:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    except Exception as e:
        if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            raise e

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device, config


def load_model_and_config_with_checkpoint(experiment_name: str, dataset: str, checkpoint_dir: str) -> Tuple[torch.nn.Module, torch.device, dict]:
    """Instantiate LSNR model and load checkpoint from specific directory."""
    config = load_experiment_config(experiment_name, dataset)
    set_seeds(config['global']['seed'])
    device = get_device(config['global']['device'])

    model_type = config['experiment']['model']
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_type}' not found in registry. Expected 'lsnr'.")
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(config['models'][model_type], config['datasets'][dataset], device)

    # Use provided checkpoint directory
    ckpt_path = latest_checkpoint(checkpoint_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")

    print(f"Loading checkpoint from {ckpt_path}")
    # PyTorch 2.6 compatibility for weights_only default
    try:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    except Exception as e:
        if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            raise e

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device, config


def sample_mind_news(news_tsv_path: str, k: int = 10) -> List[dict]:
    """Load MIND news.tsv and sample k rows; return list with id, title, abstract."""
    df = pd.read_table(
        news_tsv_path,
        names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
        usecols=['id', 'title', 'abstract']
    )
    df['title'] = df['title'].fillna('')
    df['abstract'] = df['abstract'].fillna('')
    if len(df) == 0:
        return []
    indices = random.sample(range(len(df)), min(k, len(df)))
    rows = []
    for i in indices:
        r = df.iloc[i]
        rows.append({
            'id': str(r['id']),
            'title': str(r['title']),
            'abstract': str(r['abstract'])
        })
    return rows


def tokenize_batch(texts: List[str], tokenizer, max_length: int, device: torch.device, return_encodings: bool = False):
    enc = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    batch = {
        'input_ids': enc['input_ids'].unsqueeze(1).to(device),        # [batch, 1, seq_len]
        'attention_mask': enc['attention_mask'].unsqueeze(1).to(device)  # [batch, 1, seq_len]
    }
    if return_encodings:
        return batch, enc
    return batch


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Inspect LSNR news vectors on MIND')
    parser.add_argument('--experiment', default='lsnr_baseline', help='Experiment YAML in experiments/')
    parser.add_argument('--dataset', default='mind', help='Dataset key (default: mind)')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'], help='MIND split to sample from')
    parser.add_argument('--k', type=int, default=10, help='Number of articles to inspect')
    parser.add_argument('--checkpoint-dir', help='Specific checkpoint directory to use (overrides experiment-based path)')
    parser.add_argument('--find-top-overlap', action='store_true', help='Scan entire split and report news with highest lexical overlap to sparse vector')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top-overlap articles to report when --find-top-overlap is set')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for overlap search')
    parser.add_argument('--max-sparse-tokens', type=int, default=256, help='Maximum sparse tokens (highest-weight dimensions) to consider per article when measuring overlap')
    args = parser.parse_args()

    # Load model and config
    if args.checkpoint_dir:
        model, device, config = load_model_and_config_with_checkpoint(args.experiment, args.dataset, args.checkpoint_dir)
    else:
        model, device, config = load_model_and_config(args.experiment, args.dataset)

    # Resolve tokenizer and max_length from model/dataset
    # Create tokenizer using the same model name as the PLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['datasets'][args.dataset]['pretrained_model_name'])
    max_len = config['datasets'][args.dataset]['num_words_title'] + config['datasets'][args.dataset]['num_words_abstract']

    # Resolve dataset path, special-casing if needed
    dataset_path = config['datasets'].get(args.dataset, {}).get('path', os.path.join('data', args.dataset))
    news_path = os.path.join(dataset_path, args.split, 'news.tsv')
    if not os.path.exists(news_path):
        raise FileNotFoundError(f"Missing news.tsv at {news_path}")

    # Helper utilities used by both sampling and full-overlap search
    def normalize_token(tok: str) -> str:
        norm = tok
        if norm.startswith('##'):
            norm = norm[2:]
        norm = norm.lower().strip()
        return norm

    special_ids = set(getattr(tokenizer, 'all_special_ids', []) or [])
    special_tokens = set(getattr(tokenizer, 'all_special_tokens', []) or [])

    def is_valid_token(tok: str, tid: int) -> bool:
        if tid in special_ids:
            return False
        if tok in special_tokens:
            return False
        # Filter bracketed placeholders like [unused123], [CLS], etc.
        if tok.startswith('[') and tok.endswith(']'):
            return False
        # Normalize BERT wordpiece
        norm = tok
        if norm.startswith('##'):
            norm = norm[2:]
        norm = norm.lower()
        # Filter punctuation and non-alpha
        if all(ch in string.punctuation for ch in norm):
            return False
        if not any(ch.isalpha() for ch in norm):
            return False
        if norm in STOPWORDS:
            return False
        return True

    if args.find_top_overlap:
        df = pd.read_table(
            news_path,
            names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
            usecols=['id', 'title', 'abstract']
        )
        if len(df) == 0:
            print('No news rows found in dataset.')
            return

        top_results = []
        total_rows = len(df)

        def update_top_results(entry):
            nonlocal top_results
            top_results.append(entry)
            top_results.sort(key=lambda x: (x['overlap'], x['overlap_weight']), reverse=True)
            if len(top_results) > args.top_n:
                top_results = top_results[:args.top_n]

        for start in range(0, total_rows, args.batch_size):
            end = min(start + args.batch_size, total_rows)
            chunk = df.iloc[start:end]
            texts = [f"{row.title} {row.abstract}" for row in chunk.itertuples(index=False)]
            batch, enc = tokenize_batch(texts, tokenizer, max_len, device, return_encodings=True)

            with torch.no_grad():
                news_vecs = model.get_news_vector(batch).detach().cpu()

            input_ids = enc['input_ids']
            attn_mask = enc['attention_mask']

            for offset, row in enumerate(chunk.itertuples(index=False)):
                vec = news_vecs[offset]
                k = min(args.max_sparse_tokens, vec.numel())
                vals, idxs = torch.topk(vec, k=k)
                toks = tokenizer.convert_ids_to_tokens(idxs.tolist())

                sparse_tokens = []
                token_map = {}
                for w, tid, tok in zip(vals.tolist(), idxs.tolist(), toks):
                    if w <= 0:
                        break
                    if not is_valid_token(tok, tid):
                        continue
                    norm = normalize_token(tok)
                    if not norm:
                        continue
                    sparse_tokens.append((norm, tok, tid, w))
                    if norm not in token_map or w > token_map[norm]['weight']:
                        token_map[norm] = {'token': tok, 'weight': w, 'tid': tid}

                text_ids = input_ids[offset].tolist()
                mask_vals = attn_mask[offset].tolist()
                text_toks = tokenizer.convert_ids_to_tokens(text_ids)
                text_tokens_norm = set()
                for tid, mask, tok in zip(text_ids, mask_vals, text_toks):
                    if mask == 0:
                        continue
                    if not is_valid_token(tok, tid):
                        continue
                    norm = normalize_token(tok)
                    if not norm:
                        continue
                    text_tokens_norm.add(norm)

                sparse_norms = {norm for norm, _, _, _ in sparse_tokens}
                overlap_norms = sparse_norms & text_tokens_norm

                overlap_weight = sum(token_map[n]['weight'] for n in overlap_norms)

                entry = {
                    'overlap': len(overlap_norms),
                    'overlap_weight': overlap_weight,
                    'overlap_tokens': [token_map[n]['token'] for n in sorted(overlap_norms)],
                    'overlap_weights': [token_map[n]['weight'] for n in sorted(overlap_norms)],
                    'news_id': str(row.id),
                    'title': str(row.title),
                    'abstract': str(row.abstract),
                    'sparse_tokens': sparse_tokens,
                    'total_sparse_considered': len(sparse_tokens),
                    'text_tokens': len(text_tokens_norm)
                }
                update_top_results(entry)

        print('=' * 80)
        print(f"Processed {total_rows} articles. Top {len(top_results)} by lexical overlap:")
        for rank, res in enumerate(top_results, start=1):
            print('=' * 80)
            print(f"Rank {rank}: News ID {res['news_id']} | Overlap count {res['overlap']} | Overlap weight sum {res['overlap_weight']:.4f}")
            print(f"Title: {res['title']}")
            print(f"Abstract: {res['abstract']}")
            if res['overlap'] > 0:
                print('- Overlapping tokens -')
                for tok, weight in zip(res['overlap_tokens'], res['overlap_weights']):
                    print(f"  token={tok:20s} weight={weight:.6f}")
            else:
                print('- No overlapping tokens found -')
            print('- Top sparse tokens (first 20) -')
            for norm, tok, tid, weight in res['sparse_tokens'][:20]:
                print(f"  token={tok:20s} norm={norm:20s} idx={tid:6d} weight={weight:.6f}")
        return

    # Sample news
    samples = sample_mind_news(news_path, k=args.k)
    if not samples:
        print('No news to sample.')
        return

    texts = [f"{s['title']} {s['abstract']}" for s in samples]
    batch = tokenize_batch(texts, tokenizer, max_len, device)

    with torch.no_grad():
        # get_news_vector expects dict with input tensors on device
        news_vecs = model.get_news_vector(batch)  # [batch, vocab_size]
        news_vecs = news_vecs.detach().cpu()

    # --- Semantic collapse diagnostics ---
    mat = news_vecs.numpy()
    # Sparsity (fraction of zeros)
    sparsities = (mat == 0).mean(axis=1)
    print('=' * 80)
    print(f"Avg sparsity: {sparsities.mean():.6f}  (per vec: min={sparsities.min():.6f}, max={sparsities.max():.6f})")
    # Cosine similarity matrix
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat_norm = mat / norms
    cos = mat_norm @ mat_norm.T
    # Off-diagonal stats
    n = cos.shape[0]
    off_diag = cos[~np.eye(n, dtype=bool)]
    print(f"Off-diagonal cosine: mean={off_diag.mean():.6f}, std={off_diag.std():.6f}, min={off_diag.min():.6f}, max={off_diag.max():.6f}")
    # Jaccard overlap of top-k supports
    def jaccard_topk(A: np.ndarray, B: np.ndarray, k: int) -> float:
        ia = np.argpartition(-A, k-1)[:k]
        ib = np.argpartition(-B, k-1)[:k]
        sa, sb = set(ia.tolist()), set(ib.tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / max(union, 1)
    ks = [10, 50]
    for k in ks:
        vals = []
        for i in range(len(mat)):
            for j in range(i+1, len(mat)):
                vals.append(jaccard_topk(mat[i], mat[j], k))
        if len(vals) > 0:
            arr = np.array(vals, dtype=np.float32)
            print(f"Avg Jaccard@{k} across pairs: mean={arr.mean():.6f}, std={arr.std():.6f}, max={arr.max():.6f}")

    # Map top indices to tokens
    for i, s in enumerate(samples):
        vec = news_vecs[i]
        # Collect top 30 non-stopword, non-punctuation tokens
        chosen = []
        start_k = 128
        k_step = 512
        k = start_k
        while len(chosen) < 30 and k <= vec.numel():
            vals, idxs = torch.topk(vec, k=k)
            toks = tokenizer.convert_ids_to_tokens(idxs.tolist())
            for w, tid, tok in zip(vals.tolist(), idxs.tolist(), toks):
                # Skip zero or negative weights and invalid tokens
                if w <= 0:
                    continue
                if is_valid_token(tok, tid):
                    chosen.append((w, tid, tok))
                    if len(chosen) == 30:
                        break
            k += k_step
        # Fallback: if still not enough, take whatever we have filtered unique by tid
        if len(chosen) == 0:
            vals, idxs = torch.topk(vec, k=30)
            toks = tokenizer.convert_ids_to_tokens(idxs.tolist())
            for w, tid, tok in zip(vals.tolist(), idxs.tolist(), toks):
                chosen.append((w, tid, tok))
        print('=' * 80)
        print(f"News ID: {s['id']}")
        print(f"Title: {s['title']}")
        print(f"Abstract: {s['abstract']}")
        print('- Top 30 token weights -')
        for rank, (w, tid, tok) in enumerate(chosen, start=1):
            print(f"{rank:2d}. idx={tid:6d} token={tok:20s} weight={w:.6f}")


if __name__ == '__main__':
    main()
