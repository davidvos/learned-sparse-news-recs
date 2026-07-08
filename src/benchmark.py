"""
Benchmark: Index Size & Retrieval Latency
==========================================
Measures and compares storage cost and retrieval speed of:
  - Sparse models (LSNR, STAR) → Seismic inverted index
  - Dense models (NRMS, NAML, LSTUR, NPA) → FAISS HNSW index

Usage:
    python src/benchmark.py --model_type lsnr --dataset mind --experiment_name lsnr_mind_1e-05
    python src/benchmark.py --model_type nrms --dataset mind --experiment_name nrms_mind_1e-05
    python src/benchmark.py --model_type lsnr --dataset mind --experiment_name lsnr_mind_1e-05 --output results.json
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent))

from config import load_config
from data.dataset import BehaviorsDataset
from evaluate import build_news_vectors, _behaviors_collate_fn
from utils import load_model_and_checkpoint, set_seeds, get_device

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPARSE_MODELS = {"lsnr", "star"}
DENSE_MODELS = {"nrms", "naml", "lstur", "npa"}

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

WARMUP_QUERIES = 10
SEARCH_K = 10

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark index size and retrieval latency")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=sorted(SPARSE_MODELS | DENSE_MODELS),
                        help="Model type to benchmark")
    parser.add_argument("--dataset", type=str, default="mind",
                        help="Dataset type (mind, ebnerd)")
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Experiment name (checkpoint directory name)")
    parser.add_argument("--config_file", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device preference (auto, cuda, mps, cpu)")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to benchmark on (val, test)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save results as JSON")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size override for user vector computation")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sparse Index (Seismic)
# ---------------------------------------------------------------------------

def _build_seismic_index(
    news2vector: Dict[str, Any],
    tokenizer,
    model_type: str,
    work_dir: str,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build a Seismic inverted index from sparse news vectors.

    Args:
        news2vector: Mapping from news_id → sparse vector (or tuple for LSNR).
        tokenizer: HuggingFace tokenizer for mapping vocab indices → token strings.
        model_type: 'lsnr' or 'star'.
        work_dir: Temporary directory for JSONL file.

    Returns:
        (seismic_index, stats_dict)
    """
    from seismic import SeismicIndex

    # Build vocab index → token string mapping
    vocab = tokenizer.get_vocab()  # token_str → id
    id2token = {v: k for k, v in vocab.items()}

    # Write JSONL in Seismic format
    jsonl_path = os.path.join(work_dir, "news_sparse.jsonl")
    doc_id = 0
    news_id_to_doc_id = {}
    total_nnz = 0
    num_docs = 0

    with open(jsonl_path, "w") as f:
        for news_id, vec_data in tqdm(news2vector.items(), desc="Writing Seismic JSONL"):
            if news_id == "PADDED_NEWS":
                continue

            # For LSNR, vec_data is (sparse_vec, dense_vec); for STAR it's just sparse_vec
            if model_type == "lsnr":
                sparse_vec = vec_data[0]  # (sparse, dense) tuple
            else:
                sparse_vec = vec_data

            if isinstance(sparse_vec, torch.Tensor):
                sparse_vec = sparse_vec.cpu().numpy()

            # Extract non-zero entries
            nonzero_indices = np.nonzero(sparse_vec)[0]
            if len(nonzero_indices) == 0:
                continue

            vector_dict = {}
            for idx in nonzero_indices:
                token_str = id2token.get(int(idx), f"unk_{idx}")
                weight = float(sparse_vec[idx])
                if weight > 0:
                    vector_dict[token_str] = weight

            if not vector_dict:
                continue

            doc = {"id": doc_id, "vector": vector_dict}
            f.write(json.dumps(doc) + "\n")
            news_id_to_doc_id[news_id] = doc_id
            total_nnz += len(vector_dict)
            num_docs += 1
            doc_id += 1

    # Build Seismic index
    print(f"\nBuilding Seismic index from {num_docs} documents...")
    index = SeismicIndex.build(jsonl_path)

    # Capture space usage from Seismic's built-in reporting
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    index.print_space_usage_byte()
    sys.stdout = old_stdout
    space_usage_str = captured.getvalue()
    print(space_usage_str)

    # Use Seismic's own stats
    avg_nnz = index.nnz / index.len if index.len > 0 else 0
    vocab_size = index.dim
    sparsity = 1.0 - (avg_nnz / vocab_size) if vocab_size > 0 else 0.0

    # Parse total bytes from print_space_usage_byte() output
    # Seismic typically prints lines like "Total: 12345678 bytes"
    index_size_bytes = 0
    for line in space_usage_str.strip().split("\n"):
        line_lower = line.lower().strip()
        # Look for a "total" line with a byte count
        if "total" in line_lower:
            numbers = re.findall(r'[\d,]+', line)
            if numbers:
                # Take the last (largest) number on the total line
                index_size_bytes = int(numbers[-1].replace(",", ""))
                break

    # Fallback: analytical estimate if parsing failed
    if index_size_bytes == 0:
        # Each posting ≈ 4 bytes (doc_id u16/u32) + 4 bytes (weight f32) = ~6-8 bytes
        index_size_bytes = int(index.nnz * 8)  # Conservative estimate

    stats = {
        "index_type": "seismic",
        "num_articles": index.len,
        "vector_dim": vocab_size,
        "avg_nnz_per_doc": avg_nnz,
        "avg_sparsity": sparsity,
        "index_size_bytes": index_size_bytes,
        "index_size_mb": index_size_bytes / (1024 * 1024),
        "space_usage_detail": space_usage_str.strip(),
    }

    return index, stats, news_id_to_doc_id



def _search_seismic(
    index,
    user_vectors: List[Tuple[np.ndarray, np.ndarray]],
) -> List[float]:
    """
    Measure per-query retrieval latency on a Seismic index.

    Args:
        index: SeismicIndex instance.
        user_vectors: List of (query_components, query_values) tuples,
                      where components are token strings and values are float32.

    Returns:
        List of query latencies in seconds.
    """
    from seismic import get_seismic_string
    string_type = get_seismic_string()

    latencies = []

    # Warmup
    for i in range(min(WARMUP_QUERIES, len(user_vectors))):
        components, values = user_vectors[i]
        index.search(
            query_id="warmup",
            query_components=np.array(components, dtype=string_type),
            query_values=np.array(values, dtype=np.float32),
            k=SEARCH_K,
        )

    # Timed queries
    for i, (components, values) in enumerate(tqdm(user_vectors, desc="Seismic search")):
        comp_arr = np.array(components, dtype=string_type)
        val_arr = np.array(values, dtype=np.float32)

        t0 = time.perf_counter()
        index.search(
            query_id=str(i),
            query_components=comp_arr,
            query_values=val_arr,
            k=SEARCH_K,
        )
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    return latencies


# ---------------------------------------------------------------------------
# Dense Index (FAISS HNSW)
# ---------------------------------------------------------------------------

def _build_faiss_index(
    news2vector: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build a FAISS HNSW index from dense news vectors.

    Args:
        news2vector: Mapping from news_id → dense vector (Tensor).

    Returns:
        (faiss_index, stats_dict, news_id_to_row)
    """
    import faiss

    # Collect all dense vectors
    news_ids = []
    vectors = []
    for news_id, vec in news2vector.items():
        if news_id == "PADDED_NEWS":
            continue
        if isinstance(vec, torch.Tensor):
            vec = vec.cpu().numpy()
        vectors.append(vec.astype(np.float32))
        news_ids.append(news_id)

    vectors_matrix = np.stack(vectors, axis=0)
    num_articles, dim = vectors_matrix.shape

    # Use HNSW with an inner-product metric to match the dot-product scoring
    # used by the dense recommenders.
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    print(f"\nBuilding FAISS HNSW index: {num_articles} vectors × {dim} dims, M={HNSW_M}...")
    index.add(vectors_matrix)

    # Measure index size via serialization
    serialized = faiss.serialize_index(index)
    index_size_bytes = len(serialized)

    news_id_to_row = {nid: i for i, nid in enumerate(news_ids)}

    stats = {
        "index_type": "faiss_hnsw",
        "num_articles": num_articles,
        "vector_dim": dim,
        "avg_nnz_per_doc": dim,  # Dense: all entries are non-zero
        "avg_sparsity": 0.0,
        "index_size_bytes": index_size_bytes,
        "index_size_mb": index_size_bytes / (1024 * 1024),
        "space_usage_detail": f"HNSW M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}, efSearch={HNSW_EF_SEARCH}",
    }

    return index, stats, news_id_to_row


def _search_faiss(
    index,
    user_vectors: np.ndarray,
) -> List[float]:
    """
    Measure per-query retrieval latency on a FAISS HNSW index.

    Args:
        index: FAISS index instance.
        user_vectors: numpy array of shape [num_queries, dim].

    Returns:
        List of query latencies in seconds.
    """
    latencies = []

    # Warmup
    for i in range(min(WARMUP_QUERIES, len(user_vectors))):
        index.search(user_vectors[i:i+1], SEARCH_K)

    # Timed queries
    for i in tqdm(range(len(user_vectors)), desc="FAISS search"):
        query = user_vectors[i:i+1]
        t0 = time.perf_counter()
        index.search(query, SEARCH_K)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    return latencies


# ---------------------------------------------------------------------------
# User vector computation
# ---------------------------------------------------------------------------

def _compute_user_vectors(
    model,
    config: Dict[str, Any],
    device: torch.device,
    news2vector: Dict[str, Any],
    split: str,
    batch_size: int,
    model_type: str,
) -> List[Any]:
    """
    Compute user vectors from the test/val behaviors dataset.

    Returns:
        For sparse models: list of sparse user vectors (tensors)
        For dense models: list of dense user vectors (tensors)
    """
    ds_cfg = config["datasets"][config["global"]["dataset"]]
    behaviors_file = f"{ds_cfg['path']}/{split}/behaviors_parsed.tsv"
    if not os.path.exists(behaviors_file):
        behaviors_file = f"{ds_cfg['path']}/{split}/behaviors.tsv"

    dataset = BehaviorsDataset(
        behaviors_file,
        cold_threshold=0,
        num_clicked_news_a_user=int(ds_cfg.get("num_clicked_news_a_user", 30)),
        min_history=int(config["global"].get("min_interactions", 0)),
    )

    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=config["training"].get("num_workers", 4),
        collate_fn=_behaviors_collate_fn,
    )

    user_vectors = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Computing user vectors ({split})"):
            if model_type == "star":
                # STAR model doesn't use pre-computed news vectors for user encoding
                # Skip — STAR user vectors are text-based and computed differently
                continue

            if model_type == "lsnr":
                # LSNR: user_history = (dense_vecs, sparse_vecs)
                user_history = (
                    torch.stack([
                        torch.stack([news2vector[nid][1] for nid in c])
                        for c in batch["clicked_news"]
                    ]).to(device),
                    torch.stack([
                        torch.stack([news2vector[nid][0] for nid in c])
                        for c in batch["clicked_news"]
                    ]).to(device),
                )
            else:
                # Dense models
                user_history = torch.stack([
                    torch.stack([news2vector[nid] for nid in c])
                    for c in batch["clicked_news"]
                ]).to(device)

            mask = batch["clicked_news_mask"].to(device)

            if model_type in ["lstur", "npa"]:
                user_vecs = model.get_user_vector(
                    batch["user"], user_history, mask=mask
                )
            else:
                user_vecs = model.get_user_vector(user_history, mask=mask)

            # Store individual user vectors
            for i in range(user_vecs.shape[0] if isinstance(user_vecs, torch.Tensor) else len(user_vecs)):
                uv = user_vecs[i].detach().cpu()
                user_vectors.append(uv)

    return user_vectors


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------

def _compute_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute latency statistics in milliseconds."""
    if not latencies:
        return {
            "latency_mean_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "num_queries": 0,
        }

    latencies_ms = np.array(latencies) * 1000  # seconds → milliseconds
    return {
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "latency_p99_ms": float(np.percentile(latencies_ms, 99)),
        "num_queries": len(latencies),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config
    config = load_config(args.config_file)
    if "global" not in config:
        config["global"] = {}
    config["global"]["dataset"] = args.dataset
    config["global"]["model_type"] = args.model_type

    set_seeds(config["global"].get("seed", 42))
    device = get_device(args.device)

    ds_cfg = config["datasets"][args.dataset]
    batch_size = args.batch_size or config["training"].get("batch_size", 16)

    # ── 1. Load model from checkpoint ────────────────────────────────────
    ckpt_dir = Path("./checkpoints") / args.experiment_name
    if not ckpt_dir.exists():
        candidates = [
            d for d in Path("./checkpoints").iterdir()
            if d.is_dir() and d.name.startswith(args.experiment_name)
        ]
        ckpt_dir = candidates[0] if candidates else None

    print(f"Loading model: {args.model_type} on {args.dataset}")
    model, _, _ = load_model_and_checkpoint(
        args.model_type, args.dataset, config, device,
        checkpoint_dir=str(ckpt_dir) if ckpt_dir else None,
    )
    model.eval()

    # ── 2. Encode all news articles ──────────────────────────────────────
    print(f"\nEncoding news articles ({args.split} split)...")
    news2vector, news2category = build_news_vectors(
        model, config, device, args.split
    )
    num_articles = len(news2vector) - 1  # Exclude PADDED_NEWS
    print(f"Encoded {num_articles} news articles")

    # ── 3. Build index ───────────────────────────────────────────────────
    is_sparse = args.model_type in SPARSE_MODELS

    with tempfile.TemporaryDirectory() as work_dir:
        if is_sparse:
            # Load tokenizer for vocab mapping
            tokenizer = AutoTokenizer.from_pretrained(
                ds_cfg["pretrained_model_name"]
            )

            index, index_stats, news_id_to_doc_id = _build_seismic_index(
                news2vector, tokenizer, args.model_type, work_dir,
            )
        else:
            index, index_stats, news_id_to_row = _build_faiss_index(
                news2vector,
            )

        # ── 4. Compute user vectors ──────────────────────────────────────
        print(f"\nComputing user vectors on {args.split} set...")
        user_vectors_raw = _compute_user_vectors(
            model, config, device, news2vector,
            args.split, batch_size, args.model_type,
        )
        print(f"Computed {len(user_vectors_raw)} user vectors")

        # ── 5. Measure retrieval latency ─────────────────────────────────
        if is_sparse and user_vectors_raw:
            # Convert sparse user vectors to Seismic query format
            tokenizer = AutoTokenizer.from_pretrained(
                ds_cfg["pretrained_model_name"]
            )
            vocab = tokenizer.get_vocab()
            id2token = {v: k for k, v in vocab.items()}

            seismic_queries = []
            for uv in tqdm(user_vectors_raw, desc="Converting user vectors to Seismic format"):
                if isinstance(uv, torch.Tensor):
                    uv = uv.numpy()
                nonzero_idx = np.nonzero(uv)[0]
                components = [id2token.get(int(idx), f"unk_{idx}") for idx in nonzero_idx]
                values = uv[nonzero_idx].astype(np.float32)
                if len(components) > 0:
                    seismic_queries.append((components, values))

            print(f"\nMeasuring retrieval latency ({len(seismic_queries)} queries)...")
            latencies = _search_seismic(index, seismic_queries)

        elif not is_sparse and user_vectors_raw:
            # Stack dense user vectors into numpy matrix
            user_matrix = np.stack(
                [uv.numpy().astype(np.float32) if isinstance(uv, torch.Tensor)
                 else uv.astype(np.float32)
                 for uv in user_vectors_raw],
                axis=0,
            )
            print(f"\nMeasuring retrieval latency ({len(user_matrix)} queries)...")
            latencies = _search_faiss(index, user_matrix)

        else:
            print("No user vectors computed (STAR model eval not supported in this benchmark)")
            latencies = []

    # ── 6. Compute stats and output ──────────────────────────────────────
    latency_stats = _compute_latency_stats(latencies)

    results = {
        "model": args.model_type,
        "dataset": args.dataset,
        "split": args.split,
        "experiment": args.experiment_name,
        **index_stats,
        **latency_stats,
    }

    # Print summary table
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Model:              {results['model']}")
    print(f"  Dataset:            {results['dataset']}")
    print(f"  Split:              {results['split']}")
    print(f"  Index Type:         {results['index_type']}")
    print(f"  Num Articles:       {results['num_articles']:,}")
    print(f"  Vector Dim:         {results['vector_dim']:,}")
    if is_sparse:
        print(f"  Avg NNZ/Doc:        {results['avg_nnz_per_doc']:.1f}")
        print(f"  Avg Sparsity:       {results['avg_sparsity']:.4f}")
    print(f"  Index Size:         {results['index_size_bytes']:,} bytes ({results['index_size_mb']:.2f} MB)")
    print("-" * 70)
    print(f"  Num Queries:        {results['num_queries']:,}")
    print(f"  Latency (mean):     {results['latency_mean_ms']:.3f} ms")
    print(f"  Latency (p50):      {results['latency_p50_ms']:.3f} ms")
    print(f"  Latency (p95):      {results['latency_p95_ms']:.3f} ms")
    print(f"  Latency (p99):      {results['latency_p99_ms']:.3f} ms")
    print("=" * 70)

    if results.get("space_usage_detail"):
        print(f"\nSeismic space usage detail:\n{results['space_usage_detail']}")

    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle appending to existing results file
        if output_path.exists():
            with open(output_path, "r") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                existing.append(results)
            else:
                existing = [existing, results]
        else:
            existing = [results]

        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
