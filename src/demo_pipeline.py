"""Fast End-to-End Demo Pipeline.

Provides a quick sanity-check pipeline that runs training, retrieval
evaluation, and latency benchmarking in a single execution. Hardcodes debug
hyperparameters (1 epoch, 2 batch size) for fast turnaround.

Usage::

    uv run python src/demo_pipeline.py --model_type lsnr --dataset mind
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent))

from config import load_config
from data.dataset import BaseDataset
from evaluate import build_news_vectors, eval_retrieval
from benchmark import _build_seismic_index, _search_seismic, _compute_user_vectors, _compute_latency_stats
from utils import set_seeds, get_device, load_model_and_checkpoint, log_metrics

def main():
    parser = argparse.ArgumentParser(description="Fast End-to-End Demo Pipeline (Train -> Eval -> Latency Bench)")
    parser.add_argument("--model_type", type=str, default="lsnr", help="Model type")
    parser.add_argument("--dataset", type=str, default="mind", help="Dataset name")
    parser.add_argument("--device", type=str, default="auto", help="Device preference")
    args = parser.parse_args()

    # 1. Configuration Setup (Force debug/fast parameters)
    print("=== [Step 1] Setting up configuration ===")
    config = load_config("config.yaml")
    config['global']['dataset'] = args.dataset
    config['global']['model_type'] = args.model_type
    config['global']['debug_mode'] = True  # Enable fast limits across eval and benchmark
    config['training']['num_epochs'] = 1
    config['training']['batch_size'] = 2
    config['training']['num_workers'] = 2
    
    device = get_device(args.device)
    set_seeds(config['global']['seed'])
    
    # Initialize fresh model
    print(f"Initializing {args.model_type} model...")
    model, _, _ = load_model_and_checkpoint(args.model_type, args.dataset, config, device, checkpoint_dir=None)
    
    # 2. Fast Training (10 steps)
    print("\n=== [Step 2] Launching Fast Training (10 steps) ===")
    config['_dataset'] = args.dataset
    ds_cfg = config['datasets'][args.dataset]
    mod_cfg = config['models'][args.model_type]
    train_cfg = config['training']
    
    ratio = int(mod_cfg.get('negative_sampling_ratio', config['global'].get('negative_sampling_ratio', 4)))
    dataset = BaseDataset(f"{ds_cfg['path']}/train/behaviors_parsed.tsv", f"{ds_cfg['path']}/train/news.tsv", config, ratio)
    
    dataloader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True, 
                            num_workers=train_cfg['num_workers'], drop_last=True, 
                            pin_memory=(device.type != 'mps'), collate_fn=getattr(dataset, "collate_fn", None))

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    step = 0
    t0 = time.time()
    for batch in tqdm(dataloader, desc="Training"):
        step += 1
        if len(batch.get("clicked_news", batch.get("user_input", {}).get("input_ids", []))) == 0:
            continue
        
        optimizer.zero_grad()
        out = model(batch["candidate_news"], batch["clicked_news"], batch["clicked_news_mask"])
        y_pred, reg = out if isinstance(out, tuple) else (out, None)
        loss = criterion(y_pred, torch.argmax(batch["clicked"], dim=1).to(device))
        
        if reg:
            loss += reg.get('total', 0)
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg['gradient_clip_norm'])
        optimizer.step()
        
        if step >= 10:
            break
            
    print(f"Fast training completed in {time.time() - t0:.2f} seconds.")

    # 3. Fast Evaluation
    print("\n=== [Step 3] Launching Fast Evaluation (20 samples) ===")
    model.eval()
    t_eval = time.time()
    news2vector, news2category = build_news_vectors(model, config, device, split='val')
    
    rec_metrics = eval_retrieval(
        model, config, device, news2vector, news2category, 
        cold_threshold=train_cfg['cold_threshold'], 
        split='val', 
        batch_size=train_cfg['batch_size'], 
        num_workers=train_cfg['num_workers']
    )
    print(f"Fast evaluation completed in {time.time() - t_eval:.2f} seconds.")
    print("Recommendation Metrics:")
    for k, v in rec_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 4. Fast Retrieval Latency Benchmarking
    print("\n=== [Step 4] Launching Fast Retrieval Latency Benchmark ===")
    t_bench = time.time()
    
    # Build Seismic Index
    temp_dir = tempfile.mkdtemp()
    tokenizer = AutoTokenizer.from_pretrained(ds_cfg["pretrained_model_name"])
    
    index, index_stats, news_id_to_doc_id = _build_seismic_index(
        news2vector, tokenizer, 'lsnr', temp_dir
    )
    
    # Compute sparse user query vectors
    user_vectors = _compute_user_vectors(
        model, config, device, news2vector, split='val', 
        batch_size=train_cfg['batch_size'] * 4, model_type='lsnr'
    )
    
    nnzs = []
    seismic_queries = []
    vocab = tokenizer.get_vocab()
    id2token = {v: k for k, v in vocab.items()}
    
    for uv in user_vectors:
        if isinstance(uv, torch.Tensor):
            uv = uv.numpy()
        nonzero_idx = np.nonzero(uv)[0]
        values = uv[nonzero_idx].astype(np.float32)
        nnzs.append(len(nonzero_idx))
        
        components = [id2token.get(int(idx), f"unk_{idx}") for idx in nonzero_idx]
        if len(components) > 0:
            seismic_queries.append((components, values))
            
    avg_user_nnz = float(np.mean(nnzs)) if nnzs else 0.0
    
    # Measure Seismic search latency
    print(f"Measuring Seismic index search latency with {len(seismic_queries)} queries...")
    latencies = _search_seismic(index, seismic_queries)
    latency_stats = _compute_latency_stats(latencies)
    
    print(f"Fast benchmarking completed in {time.time() - t_bench:.2f} seconds.")
    
    print("\n" + "="*50)
    print("DEMO PIPELINE SUMMARY REPORT")
    print("="*50)
    print(f"Model Type:            {args.model_type}")
    print(f"Dataset:               {args.dataset}")
    print(f"Avg User NNZ:          {avg_user_nnz:.2f}")
    print(f"Validation AUC:        {rec_metrics.get('ret_auc', 0.0):.4f}")
    print(f"Validation NDCG@10:    {rec_metrics.get('ret_ndcg10', 0.0):.4f}")
    print(f"Mean Search Latency:   {latency_stats['latency_mean_ms']:.3f} ms")
    print(f"P95 Search Latency:    {latency_stats['latency_p95_ms']:.3f} ms")
    print("="*50)

if __name__ == "__main__":
    main()
