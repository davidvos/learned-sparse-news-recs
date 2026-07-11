"""Detailed profile inspection for a single val row.

Note:
    Requires an existing LSNR checkpoint at the hardcoded path.

Usage::
    python src/inspect_user.py --row_idx 10478
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent))
from config import load_config
from evaluate import build_news_vectors
from utils import set_seeds, get_device, load_model_and_checkpoint

CHECKPOINT = (
    "lsnr_attention_pe_notopk_low_reg_lsnr_mind_cold0_lr1e-05_ln0.005"
    "_lh0.0005_lu0.0_hls1_attention_pe1"
)
DATASET = "mind"
MODEL_TYPE = "lsnr"

# Generic tokens identified in previous runs (superset for robustness)
HAND_GENERIC = {
    "a", "i", "to", "out", "we", "off", "us", "go", "best", "black", "white",
    "top", "power", "help", "red", "david", "green", "action", "library",
    "teachers", "teacher", "library", "anger", "dad", "##stein", "inspiration",
    "taught", "literacy", "courses", "classes", "class", "course", "families",
    "moral", "david", "ali", "liberal",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row_idx", type=int, required=True)
    parser.add_argument("--topk_show", type=int, default=20)
    args = parser.parse_args()

    config = load_config("config.yaml")
    config["global"]["dataset"] = DATASET
    config["global"]["model_type"] = MODEL_TYPE
    config["global"]["debug_mode"] = False
    config["models"]["lsnr"]["topk_sparsity_queries"] = None  # no top-k at inference

    device = get_device("auto")
    set_seeds(config["global"]["seed"])
    ds_cfg = config["datasets"][DATASET]

    print("Loading news metadata...")
    news_df = pd.read_table(
        f"{ds_cfg['path']}/val/news.tsv", header=None,
        names=["id", "category", "subcategory", "title", "abstract", "url", "te", "ae"],
        usecols=["id", "category", "title", "abstract"],
    )
    news_df["title"] = news_df["title"].fillna("")
    news_df["abstract"] = news_df["abstract"].fillna("")
    news_meta = {
        row.id: {"cat": row.category.lower(), "title": row.title, "abstract": row.abstract}
        for row in news_df.itertuples()
    }

    print("Loading behaviors...")
    beh_df = pd.read_table(
        f"{ds_cfg['path']}/val/behaviors_parsed.tsv",
        names=["impression_id", "user", "time", "clicked_news",
               "impressions", "candidate_news", "clicked"],
        usecols=range(7),
    )
    beh_df["clicked_news"] = beh_df["clicked_news"].fillna("").str.split()

    row = beh_df.iloc[args.row_idx]
    hist = row["clicked_news"]
    print(f"\nRow {args.row_idx}: user={row['user']}, hist_len={len(hist)}")

    print("\nFull click history:")
    for i, nid in enumerate(hist, 1):
        meta = news_meta.get(nid, {})
        print(f"  [{i}] cat={meta.get('cat','?')}  {meta.get('title','?')[:100]}")

    print("\nLoading model...")
    model, _, _ = load_model_and_checkpoint(
        MODEL_TYPE, DATASET, config, device,
        checkpoint_dir=f"./checkpoints/{CHECKPOINT}",
    )
    model.eval()

    print("Building news vectors...")
    news2vector, _ = build_news_vectors(model, config, device, "val")

    tokenizer = AutoTokenizer.from_pretrained(ds_cfg["pretrained_model_name"])
    id2token = {v: k for k, v in tokenizer.get_vocab().items()}

    num_news = int(ds_cfg.get("num_clicked_news_a_user", 50))
    hist_valid = [nid for nid in hist if nid in news2vector]
    hist_padded = hist_valid[-num_news:]
    pad_len = num_news - len(hist_padded)
    mask = [1] * len(hist_padded) + [0] * pad_len
    hist_padded = hist_padded + ["PADDED_NEWS"] * pad_len

    with torch.no_grad():
        user_history = (
            torch.stack([news2vector[nid][1] for nid in hist_padded]).unsqueeze(0).to(device),
            torch.stack([news2vector[nid][0] for nid in hist_padded]).unsqueeze(0).to(device),
        )
        mask_t = torch.tensor([mask], dtype=torch.float).to(device)
        uv = model.get_user_vector(user_history, mask=mask_t)
        uv = uv[0].detach().cpu().numpy()

    click_texts = [
        news_meta.get(nid, {}).get("title", "") + " " + news_meta.get(nid, {}).get("abstract", "")
        for nid in hist_valid
    ]
    click_text_lower = " ".join(click_texts).lower()

    nz = np.nonzero(uv)[0]
    vals = uv[nz]
    order = np.argsort(-vals)

    # Apply generic filter
    filtered = [
        (nz[i], vals[i]) for i in order
        if id2token.get(int(nz[i]), "") not in HAND_GENERIC
    ]

    top_idx = [i for i, _ in filtered[:args.topk_show]]
    top_vals = [v for _, v in filtered[:args.topk_show]]
    top_tokens = [id2token.get(int(i), f"unk_{i}") for i in top_idx]

    # Normalize weights to [0,1] relative to max
    max_w = max(top_vals) if top_vals else 1.0

    print(f"\nTop-{args.topk_show} profile (after generic filter, max-normalized):")
    print(f"  {'token':<20} {'raw':>8} {'norm':>8} {'in_click':>10}")
    for tok, raw, idx_val in zip(top_tokens, top_vals, top_idx):
        norm = raw / max_w
        in_click = tok.lower() in click_text_lower or tok.lower().lstrip("#") in click_text_lower
        print(f"  {tok:<20} {raw:>8.4f} {norm:>8.4f}   {'YES' if in_click else 'no'}")

    # Build normalized weights (matching original paper scale)
    norm_vals = [v / max_w for v in top_vals]

    print("\nLaTeX profile line:")
    parts = []
    for tok, nw in zip(top_tokens, norm_vals):
        in_click = tok.lower() in click_text_lower or tok.lower().lstrip("#") in click_text_lower
        entry = f"\\textbf{{{tok}}}\\,({nw:.2f})" if in_click else f"{tok}\\,({nw:.2f})"
        parts.append(entry)
    print(", ".join(parts))


if __name__ == "__main__":
    main()
