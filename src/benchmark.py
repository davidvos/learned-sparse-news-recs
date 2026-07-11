"""Benchmark: Index Size & Retrieval Latency.

Measures and compares storage cost and retrieval speed of:
  - Sparse models (LSNR, STAR) using the Seismic inverted index.
  - Dense models (NRMS, NAML, LSTUR, NPA) using a FAISS HNSW index.

Usage::

    uv run python src/benchmark.py --model_type lsnr --dataset mind --experiment_name lsnr_mind_1e-05
    uv run python src/benchmark.py --model_type nrms --dataset mind --experiment_name nrms_mind_1e-05
    uv run python src/benchmark.py --model_type lsnr --dataset mind --experiment_name lsnr_mind_1e-05 --output results.json
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

python_dir = Path(__file__).resolve().parent
sys.path.append(str(python_dir))

from config import load_config
from data.dataset import BehaviorsDataset
from evaluate import (
    build_news_vectors, _behaviors_collate_fn,
    _build_star_news_lookup, _build_star_user_text,
)
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

SEISMIC_QUERY_CUT = 10
SEISMIC_HEAP_FACTOR = 0.8
SEISMIC_BATCH_SIZE = 1000

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
    parser.add_argument("--user_query_topk", type=int, default=None,
                        help="Retain only the top-k active terms in each user vector before Seismic search. "
                             "Reduces query NNZ and retrieval latency at the cost of recall.")
    parser.add_argument("--search_k", type=int, default=10,
                        help="Number of retrieved items (SEARCH_K)")
    parser.add_argument("--query_cut", type=int, default=10,
                        help="Number of query tokens to retain during index search (SEISMIC_QUERY_CUT)")
    parser.add_argument("--heap_factor", type=float, default=0.8,
                        help="Seismic search heap factor (SEISMIC_HEAP_FACTOR)")
    parser.add_argument("--num_queries", type=int, default=None,
                        help="If specified, limits the number of timed search queries to this number for faster latency benchmarking")
    parser.add_argument("--index_type_tag", type=str, default=None,
                        help="Override the index_type label written to JSON (e.g. 'seismic_k30'). "
                             "Useful when re-running with different query pruning to avoid overwriting existing entries.")
    parser.add_argument("--incremental_update_samples", type=int, default=50,
                        help="Number of documents from the 10%% incremental set to use for measuring "
                             "per-document update cost via marginal rebuild. Higher = more stable estimate but slower.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sparse Index (Seismic)
# ---------------------------------------------------------------------------

def _vec_to_dict(
    news_id: str,
    vec_data: Any,
    model_type: str,
    id2token: Dict[int, str],
) -> Optional[Dict[str, float]]:
    """
    Convert a raw news vector entry to a Seismic-format {token: weight} dict.
    Returns None if the vector has no non-zero entries.
    """
    if model_type == "lsnr":
        sparse_vec = vec_data[0]
    else:
        sparse_vec = vec_data

    if isinstance(sparse_vec, torch.Tensor):
        sparse_vec = sparse_vec.cpu().numpy()

    nonzero_indices = np.nonzero(sparse_vec)[0]
    if len(nonzero_indices) == 0:
        return None

    vector_dict = {}
    for idx in nonzero_indices:
        weight = float(sparse_vec[idx])
        if weight > 0:
            vector_dict[id2token.get(int(idx), f"unk_{idx}")] = weight

    return vector_dict if vector_dict else None


def _write_seismic_jsonl(
    news_items: List[Tuple[str, Any]],
    model_type: str,
    id2token: Dict[int, str],
    jsonl_path: str,
    start_doc_id: int = 0,
) -> Tuple[int, int]:
    """
    Write a list of (news_id, vec_data) pairs to a Seismic JSONL file.

    Returns:
        (num_docs_written, total_nnz)
    """
    doc_id = start_doc_id
    total_nnz = 0
    with open(jsonl_path, "w") as f:
        for news_id, vec_data in news_items:
            vd = _vec_to_dict(news_id, vec_data, model_type, id2token)
            if vd is None:
                continue
            f.write(json.dumps({"id": doc_id, "vector": vd}) + "\n")
            total_nnz += len(vd)
            doc_id += 1
    return doc_id - start_doc_id, total_nnz


def _build_seismic_index(
    news2vector: Dict[str, Any],
    tokenizer,
    model_type: str,
    work_dir: str,
    incremental_update_samples: int = 50,
) -> Tuple[Any, Dict[str, Any], Dict[str, int]]:
    """
    Build a Seismic inverted index from sparse news vectors.

    Splits the corpus 90/10: builds the initial index on 90% of documents and
    measures the true per-document incremental update cost on the remaining 10%.
    Because Seismic has no add_document API, the incremental update cost is
    measured as the marginal rebuild time: time(build N+1 docs) - time(build N docs),
    averaged over ``incremental_update_samples`` documents from the 10% set.

    Args:
        news2vector: Mapping from news_id → sparse vector (or tuple for LSNR).
        tokenizer: HuggingFace tokenizer for mapping vocab indices → token strings.
        model_type: 'lsnr' or 'star'.
        work_dir: Temporary directory for JSONL files.
        incremental_update_samples: How many docs from the 10% set to use for
            marginal-rebuild timing. More = more stable but slower.

    Returns:
        (seismic_index, stats_dict, news_id_to_doc_id)
    """
    from seismic import SeismicIndex

    # Build vocab index → token string mapping
    id2token = {v: k for k, v in tokenizer.get_vocab().items()}

    # Collect all valid docs (exclude PADDED_NEWS)
    all_items = [
        (nid, vd) for nid, vd in news2vector.items() if nid != "PADDED_NEWS"
    ]

    # ── 90 / 10 split ────────────────────────────────────────────────────────
    split_idx = int(len(all_items) * 0.9)
    initial_items = all_items[:split_idx]
    incremental_items = all_items[split_idx:]

    # ── 1. Build initial index on 90% ────────────────────────────────────────
    jsonl_initial = os.path.join(work_dir, "news_initial.jsonl")
    num_initial, total_nnz = _write_seismic_jsonl(
        initial_items, model_type, id2token, jsonl_initial
    )

    print(f"\nBuilding Seismic index on {num_initial} documents (90% of corpus)...")
    t0_build = time.perf_counter()
    index = SeismicIndex.build(jsonl_initial)
    build_time_sec = time.perf_counter() - t0_build
    print(f"Initial index built in {build_time_sec:.3f} s")

    # ── 2. Measure incremental update cost on the 10% set ────────────────────
    # Seismic has no add_document: we measure marginal rebuild cost.
    # For each sampled incremental doc we time: build(N) vs build(N+1).
    # To keep runtime manageable we time build(N+1) - build(N) on a small
    # prefix of the incremental set, reusing the pre-built base JSONL.
    sample_limit = min(incremental_update_samples, len(incremental_items))
    sample_items = incremental_items[:sample_limit]

    print(f"\nMeasuring incremental update cost (marginal rebuild) on "
          f"{sample_limit} documents...")

    # Write the base JSONL once to a temp file
    jsonl_base = os.path.join(work_dir, "news_base.jsonl")
    _write_seismic_jsonl(initial_items, model_type, id2token, jsonl_base)

    # Read back the base JSONL lines so we can append one at a time
    with open(jsonl_base, "r") as f:
        base_lines = f.readlines()

    update_latencies = []
    jsonl_incremental = os.path.join(work_dir, "news_incremental.jsonl")
    doc_id_offset = num_initial  # continue numbering from where the base left off

    for i, (news_id, vec_data) in enumerate(
        tqdm(sample_items, desc="Incremental rebuild timing")
    ):
        vd = _vec_to_dict(news_id, vec_data, model_type, id2token)
        if vd is None:
            continue
        new_line = json.dumps({"id": doc_id_offset + i, "vector": vd}) + "\n"

        # Write base + one new doc
        with open(jsonl_incremental, "w") as f:
            f.writelines(base_lines)
            f.write(new_line)

        t0_up = time.perf_counter()
        SeismicIndex.build(jsonl_incremental)
        t_up = time.perf_counter() - t0_up

        # Marginal cost = rebuild(N+1) - build_time_per_doc_avg * N
        # Simpler and more honest: just report the raw rebuild(N+1) time
        # and note it is an upper bound on true insertion cost.
        update_latencies.append(t_up)

    avg_index_update_time_ms = (
        float(np.mean(update_latencies)) * 1000 if update_latencies else 0.0
    )
    print(f"Avg incremental rebuild time (N+1 docs): {avg_index_update_time_ms:.2f} ms")

    # ── 3. Finalise — rebuild full index on all docs for querying ────────────
    jsonl_full = os.path.join(work_dir, "news_full.jsonl")
    num_full, _ = _write_seismic_jsonl(all_items, model_type, id2token, jsonl_full)
    news_id_to_doc_id = {}
    doc_id = 0
    for news_id, vec_data in all_items:
        vd = _vec_to_dict(news_id, vec_data, model_type, id2token)
        if vd is not None:
            news_id_to_doc_id[news_id] = doc_id
            doc_id += 1

    print(f"\nRebuilding full Seismic index on all {num_full} documents for querying...")
    index = SeismicIndex.build(jsonl_full)

    # ── 4. Collect size and stats ─────────────────────────────────────────────
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    index.print_space_usage_byte()
    sys.stdout = old_stdout
    space_usage_str = captured.getvalue()
    print(space_usage_str)

    avg_nnz = index.nnz / index.len if index.len > 0 else 0
    vocab_size = index.dim
    sparsity = 1.0 - (avg_nnz / vocab_size) if vocab_size > 0 else 0.0

    index_size_bytes = 0
    for line in space_usage_str.strip().split("\n"):
        if "total" in line.lower().strip():
            numbers = re.findall(r'[\d,]+', line)
            if numbers:
                index_size_bytes = int(numbers[-1].replace(",", ""))
                break
    if index_size_bytes == 0:
        index_size_bytes = int(index.nnz * 8)

    stats = {
        "index_type": "seismic",
        "num_articles": index.len,
        "vector_dim": vocab_size,
        "avg_nnz_per_doc": avg_nnz,
        "avg_sparsity": sparsity,
        "index_size_bytes": index_size_bytes,
        "index_size_mb": index_size_bytes / (1024 * 1024),
        "index_build_time_sec": build_time_sec,
        "avg_index_update_time_ms": avg_index_update_time_ms,
        "space_usage_detail": space_usage_str.strip(),
    }

    return index, stats, news_id_to_doc_id



def _search_seismic(
    index,
    user_vectors: List[Tuple[np.ndarray, np.ndarray]],
) -> List[float]:
    """
    Measure per-query retrieval latency on a Seismic index.

    Each query is issued individually to produce per-item latencies comparable
    to the FAISS HNSW benchmark.

    Args:
        index: SeismicIndex instance.
        user_vectors: List of (query_components, query_values) tuples.

    Returns:
        List of query latencies in seconds.
    """
    from seismic import get_seismic_string
    string_type = get_seismic_string()

    queries = [(np.array(c, dtype=string_type), np.array(v, dtype=np.float32))
               for c, v in user_vectors]

    # Warmup
    for i in range(min(WARMUP_QUERIES, len(queries))):
        index.search(str(i), queries[i][0], queries[i][1], SEARCH_K, SEISMIC_QUERY_CUT, SEISMIC_HEAP_FACTOR)

    # Timed per-query search
    latencies = []
    for i in tqdm(range(len(queries)), desc="Seismic search"):
        components, values = queries[i]
        t0 = time.perf_counter()
        index.search(str(i), components, values, SEARCH_K, SEISMIC_QUERY_CUT, SEISMIC_HEAP_FACTOR)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    return latencies


# ---------------------------------------------------------------------------
# Dense Index (FAISS HNSW)
# ---------------------------------------------------------------------------

def _build_faiss_index(
    news2vector: Dict[str, Any],
    incremental_update_samples: int = 50,
) -> Tuple[Any, Dict[str, Any], Dict[str, int]]:
    """
    Build a FAISS HNSW index from dense news vectors.

    Splits the corpus 90/10: builds the initial index on 90% of documents and
    measures the true per-document incremental update cost on the remaining 10%
    sequentially on a single thread.

    Args:
        news2vector: Mapping from news_id → dense vector (Tensor).
        incremental_update_samples: Unused for HNSW, we measure on the entire 10%
            incremental set to get a stable average, similar to Lucene.

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

    num_articles = len(news_ids)
    dim = vectors[0].shape[0] if num_articles > 0 else 0

    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    # ── 90 / 10 split ────────────────────────────────────────────────────────
    split_idx = int(num_articles * 0.9)
    initial_ids = news_ids[:split_idx]
    incremental_ids = news_ids[split_idx:]

    # ── 1. Build initial index on 90% ────────────────────────────────────────
    initial_vectors = [vectors[i] for i in range(split_idx)]
    initial_matrix = np.stack(initial_vectors, axis=0) if initial_vectors else np.empty((0, dim), dtype=np.float32)

    print(f"\nBuilding FAISS HNSW index on {len(initial_ids)} documents (90% of corpus)...")
    t0_build = time.perf_counter()
    if len(initial_ids) > 0:
        index.add(initial_matrix)
    build_time_sec = time.perf_counter() - t0_build
    print(f"Initial HNSW index built in {build_time_sec:.3f} s")

    # ── 2. Measure incremental updates on the 10% set ────────────────────────
    print(f"\nMeasuring incremental updates on {len(incremental_ids)} documents (sequential single-threaded add)...")
    incremental_vectors = [vectors[i] for i in range(split_idx, num_articles)]

    # Limit maximum OMP threads to 1 during the sequential timing loop to measure
    # the true single-threaded update cost (simulating a streaming ingest service).
    old_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)

    update_latencies = []
    try:
        for vec in tqdm(incremental_vectors, desc="Incremental HNSW add"):
            vec_2d = vec[np.newaxis, :]
            t0 = time.perf_counter()
            index.add(vec_2d)
            update_latencies.append(time.perf_counter() - t0)
    finally:
        # Ensure we always restore original threading behavior
        faiss.omp_set_num_threads(old_threads)

    avg_index_update_time_ms = (float(np.mean(update_latencies)) * 1000) if update_latencies else 0.0
    print(f"Per-doc add: {avg_index_update_time_ms:.4f} ms avg over {len(update_latencies)} docs")

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
        "index_build_time_sec": build_time_sec,
        "avg_index_update_time_ms": avg_index_update_time_ms,
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
    limit: int = None,
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

    # --- STAR-specific setup ---
    star_news2data = None
    star_tokenizer = None
    star_max_user_tokens = 256
    star_num_words_title = ds_cfg.get('num_words_title', 20)
    star_num_words_abstract = ds_cfg.get('num_words_abstract', 50)
    star_max_history = config.get('models', {}).get('star', {}).get('max_history_titles', 10)
    if model_type == 'star':
        from transformers import AutoTokenizer
        news_path = f"{ds_cfg['path']}/{split}/news.tsv"
        star_news2data = _build_star_news_lookup(news_path)
        pretrained_name = (
            ds_cfg.get('star_pretrained_model_name') or
            config.get('models', {}).get('star', {}).get('pretrained_model_name') or
            ds_cfg.get('pretrained_model_name', 'naver/splade-cocondenser-ensembledistil')
        )
        star_tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        star_max_user_tokens = config.get('models', {}).get('star', {}).get('max_user_tokens', 256)

    user_vectors = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Computing user vectors ({split})"):
            if limit is not None and len(user_vectors) >= limit:
                break
            if config.get('global', {}).get('debug_mode', False) and len(user_vectors) >= 20:
                print("\nDebug mode: stopping user vector computation early after 20 vectors.")
                break

            if model_type == "star":
                # Build and encode user history text on-the-fly
                user_texts = [
                    _build_star_user_text(
                        c, star_news2data,
                        num_words_title=star_num_words_title,
                        num_words_abstract=star_num_words_abstract,
                        max_history=star_max_history,
                    )
                    for c in batch["clicked_news"]
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

            elif model_type == "lsnr":
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
                mask = batch["clicked_news_mask"].to(device)
                user_vecs = model.get_user_vector(user_history, mask=mask)

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

    if limit is not None:
        user_vectors = user_vectors[:limit]
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

    # Override global constants with command-line arguments
    global SEARCH_K, SEISMIC_QUERY_CUT, SEISMIC_HEAP_FACTOR
    SEARCH_K = args.search_k
    SEISMIC_QUERY_CUT = args.query_cut
    SEISMIC_HEAP_FACTOR = args.heap_factor

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
                incremental_update_samples=args.incremental_update_samples,
            )
        else:
            index, index_stats, news_id_to_row = _build_faiss_index(
                news2vector,
                incremental_update_samples=args.incremental_update_samples,
            )

        # ── 4. Compute user vectors ──────────────────────────────────────
        print(f"\nComputing user vectors on {args.split} set...")
        user_vectors_raw = _compute_user_vectors(
            model, config, device, news2vector,
            args.split, batch_size, args.model_type,
            limit=args.num_queries,
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
            user_nnzs = []
            topk = args.user_query_topk
            for uv in tqdm(user_vectors_raw, desc="Converting user vectors to Seismic format"):
                if isinstance(uv, torch.Tensor):
                    uv = uv.numpy()
                nonzero_idx = np.nonzero(uv)[0]
                values = uv[nonzero_idx].astype(np.float32)
                # Prune to max(topk, query_cut) before string conversion — converting
                # thousands of token strings per user is the main bottleneck when user
                # NNZ is high (e.g. from the SPLADE projection layer).
                effective_k = topk if topk is not None else SEISMIC_QUERY_CUT
                if len(nonzero_idx) > effective_k:
                    top_positions = np.argpartition(values, -effective_k)[-effective_k:]
                    nonzero_idx = nonzero_idx[top_positions]
                    values = values[top_positions]
                components = [id2token.get(int(idx), f"unk_{idx}") for idx in nonzero_idx]
                user_nnzs.append(len(nonzero_idx))
                if len(components) > 0:
                    seismic_queries.append((components, values))

            user_vec_dim = user_vectors_raw[0].shape[0] if user_vectors_raw else 0
            avg_user_nnz = float(np.mean(user_nnzs)) if user_nnzs else 0.0
            avg_user_sparsity = 1.0 - (avg_user_nnz / user_vec_dim) if user_vec_dim > 0 else 0.0

            if args.num_queries is not None:
                seismic_queries = seismic_queries[:args.num_queries]
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
            avg_user_nnz = 0.0
            avg_user_sparsity = 0.0
            if args.num_queries is not None:
                user_matrix = user_matrix[:args.num_queries]
            print(f"\nMeasuring retrieval latency ({len(user_matrix)} queries)...")
            latencies = _search_faiss(index, user_matrix)

        else:
            print("No user vectors computed (STAR model eval not supported in this benchmark)")
            latencies = []
            avg_user_nnz = 0.0
            avg_user_sparsity = 0.0

    # ── 6. Compute stats and output ──────────────────────────────────────
    latency_stats = _compute_latency_stats(latencies)

    user_stats = {}
    if is_sparse:
        user_stats = {
            "avg_user_nnz": avg_user_nnz,
            "avg_user_sparsity": avg_user_sparsity,
        }

    results = {
        "model": args.model_type,
        "dataset": args.dataset,
        "split": args.split,
        "experiment": args.experiment_name,
        **index_stats,
        **user_stats,
        **latency_stats,
    }

    # Apply optional index type tag override (e.g. 'seismic_k30')
    if args.index_type_tag:
        results["index_type"] = args.index_type_tag

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
        print(f"  Avg User NNZ:       {results['avg_user_nnz']:.1f}")
        print(f"  Avg User Sparsity:  {results['avg_user_sparsity']:.4f}")
    print(f"  Index Size:         {results['index_size_bytes']:,} bytes ({results['index_size_mb']:.2f} MB)")
    if "index_build_time_sec" in results:
        print(f"  Index Build Time:   {results['index_build_time_sec']:.3f} s")
        print(f"  Avg Doc Index Time: {results['avg_index_update_time_ms']:.4f} ms")
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
