#!/usr/bin/env python3
"""Compare LSNR and NRMS hit/fail patterns and lexical overlap."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import load_experiment_config
from evaluate import MODEL_REGISTRY, build_news_vectors
from utils import get_device, latest_checkpoint, set_seeds

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a comparison experiment between LSNR and NRMS to analyse hit/fail "
            "patterns and lexical overlap of the top-ranked news items."
        )
    )
    parser.add_argument(
        "--lsnr-experiment",
        default="lsnr_baseline",
        help="Experiment name for LSNR (YAML in experiments/).",
    )
    parser.add_argument(
        "--nrms-experiment",
        default="nrms_baseline",
        help="Experiment name for NRMS (YAML in experiments/).",
    )
    parser.add_argument(
        "--dataset",
        default="mind",
        help="Dataset key to evaluate (must exist in experiment config).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--cold-threshold",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "One or more cold thresholds identifying behaviors_parsed_coldX.tsv files to analyse. "
            "Pass multiple values to compare several thresholds in a single run."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Batch size for building user vectors during evaluation (independent of "
            "training batch sizes)."
        ),
    )
    parser.add_argument(
        "--news-device",
        choices=["auto", "cuda", "mps", "cpu"],
        help=(
            "Optional device override used while encoding news vectors. "
            "Defaults to the model device; specify 'mps' to attempt Apple GPU acceleration."
        ),
    )
    parser.add_argument(
        "--max-impressions",
        type=int,
        default=None,
        help="Limit the number of impressions analysed (for quick smoke tests).",
    )
    parser.add_argument(
        "--lsnr-checkpoint-dir",
        help="Optional explicit checkpoint directory for LSNR (overrides default lookup).",
    )
    parser.add_argument(
        "--nrms-checkpoint-dir",
        help="Optional explicit checkpoint directory for NRMS (overrides default lookup).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save per-impression analysis as CSV.",
    )
    parser.add_argument(
        "--overlap-sample-count",
        type=int,
        default=0,
        help=(
            "If >0, print this many lowest overlap-delta samples for lsnr_only and"
            " nrms_only buckets."
        ),
    )
    return parser.parse_args()


def load_model(
    experiment_name: str,
    dataset: str,
    checkpoint_dir: str | None = None,
):
    config = load_experiment_config(experiment_name, dataset)
    set_seeds(int(config["global"]["seed"]))
    device = get_device(config["global"]["device"])

    model_key = str(config["experiment"]["model"])
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_key}' is not registered.")
    model_class = MODEL_REGISTRY[model_key]
    model = model_class(config["models"][model_key], config["datasets"][dataset], device)

    lr = config["training"].get("learning_rate", 1e-5)
    cold_threshold = config["training"].get("cold_threshold", 0)

    def format_lr(value: float) -> str:
        # Preserve scientific notation if that is how checkpoints are named.
        if value == 0:
            return "0"
        return f"{value:.0e}" if value < 1e-3 else str(value)

    lr_str = format_lr(float(lr))
    base_name = f"{experiment_name}_{model_key}_{dataset}_cold{cold_threshold}_lr{lr_str}"

    expected_names: list[str] = [base_name]
    if model_key == "lsnr":
        # Backward compatibility: older checkpoints were suffixed with _idf/_noidf.
        expected_names.extend([base_name + "_idf", base_name + "_noidf"])
    else:
        expected_names.append(base_name + "_noidf")

    checkpoints_root = REPO_ROOT / "checkpoints"

    def resolve_default_dir() -> Path | None:
        for name in expected_names:
            candidate = checkpoints_root / name
            if candidate.exists():
                return candidate
        return None

    if checkpoint_dir is None:
        resolved = resolve_default_dir()
    else:
        candidate = Path(checkpoint_dir)
        if not candidate.exists():
            alt = checkpoints_root / candidate
            candidate = alt if alt.exists() else candidate
        resolved = candidate if candidate.exists() else None
        if resolved is None:
            fallback = resolve_default_dir()
            if fallback is not None:
                resolved = fallback

    if resolved is None:
        prefix = f"{experiment_name}_{model_key}_{dataset}_cold{cold_threshold}_lr"
        if checkpoints_root.exists():
            candidates = sorted(
                p.name for p in checkpoints_root.iterdir() if p.is_dir() and p.name.startswith(prefix)
            )
        else:
            candidates = []

        if len(candidates) == 1:
            resolved = checkpoints_root / candidates[0]
            print(
                "Warning: using checkpoint directory",
                resolved,
                "(closest match; learning rate differs from config)",
                flush=True,
            )
        else:
            hint = f" Available candidates: {candidates}" if candidates else ""
            raise FileNotFoundError(
                f"No checkpoint directory matched any of {expected_names}.{hint}"
            )

    checkpoint_dir = resolved

    ckpt_path = latest_checkpoint(str(checkpoint_dir))
    if ckpt_path is None:
        prefix = f"{experiment_name}_{model_key}_{dataset}_cold{cold_threshold}"  # reuse from above
        if checkpoints_root.exists():
            candidates = sorted(
                p.name for p in checkpoints_root.iterdir() if p.is_dir() and p.name.startswith(prefix)
            )
        else:
            candidates = []
        hint = f" Available candidates: {candidates}" if candidates else ""
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}.{hint}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    except Exception as exc:
        if "weights_only" in str(exc) or "WeightsUnpickler" in str(exc):
            checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
        else:
            raise

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device, config


def tokenize_text(text: str) -> List[str]:
    # Use Unicode-aware word tokenisation to keep multilingual tokens.
    return re.findall(r"\w+", str(text).lower())


def build_news_token_map(dataset_path: Path, splits: Sequence[str]) -> Dict[str, List[str]]:
    token_map: Dict[str, List[str]] = {}
    for split in splits:
        news_file = dataset_path / split / "news.tsv"
        if not news_file.exists():
            continue
        df = pd.read_table(
            news_file,
            names=[
                "id",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ],
            usecols=["id", "title", "abstract"],
        )
        df["title"] = df["title"].fillna("")
        df["abstract"] = df["abstract"].fillna("")
        for row in df.itertuples(index=False):
            combined = f"{row.title} {row.abstract}".strip()
            token_map[str(row.id)] = tokenize_text(combined)
    return token_map


def load_behaviors(
    behaviors_file: Path,
    max_impressions: int | None = None,
) -> List[dict]:
    df = pd.read_table(
        behaviors_file,
        header=None,
        usecols=range(5),
        names=["impression_id", "user", "time", "clicked_news", "impressions"],
    )

    if max_impressions is not None:
        df = df.head(max_impressions)

    samples: List[dict] = []
    for row in df.itertuples(index=False):
        clicked_news = row.clicked_news.split() if isinstance(row.clicked_news, str) else []
        candidate_pairs = []
        candidate_ids: List[str] = []
        labels: List[int] = []
        for token in str(row.impressions).split():
            if "-" not in token:
                continue
            news_id, label = token.rsplit("-", 1)
            if news_id == "PADDED_NEWS":
                continue
            candidate_pairs.append((news_id, int(label)))
            candidate_ids.append(news_id)
            labels.append(int(label))

        if not candidate_ids:
            continue
        positive_count = sum(labels)
        if positive_count == 0 or positive_count == len(labels):
            # Match evaluation script behaviour: skip degenerate cases.
            continue

        samples.append(
            {
                "impression_id": str(row.impression_id),
                "user": str(row.user),
                "clicked_news": clicked_news,
                "candidate_ids": candidate_ids,
                "labels": labels,
            }
        )
    return samples


def pad_histories(
    histories: Sequence[Sequence[str]],
    pad_token: str,
) -> List[List[str]]:
    max_len = max((len(h) for h in histories), default=0)
    max_len = max(1, max_len)  # ensure non-zero for empty histories
    padded = []
    for history in histories:
        pad_count = max_len - len(history)
        padded.append([pad_token] * pad_count + list(history))
    return padded


def collect_predictions(
    model,
    model_type: str,
    device: torch.device,
    samples: Sequence[dict],
    news2vector: Dict[str, torch.Tensor],
    batch_size: int,
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    pad_vector = news2vector.get("PADDED_NEWS")
    if pad_vector is None:
        first_vector = next(iter(news2vector.values()))
        pad_vector = torch.zeros_like(first_vector)

    with torch.no_grad():
        iterator = range(0, len(samples), batch_size)
        for start in tqdm(
            iterator,
            desc=f"{model_type.upper()} scoring",
            unit="batch",
            leave=False,
        ):
            chunk = samples[start : start + batch_size]
            histories = [sample["clicked_news"] for sample in chunk]
            padded_histories = pad_histories(histories, "PADDED_NEWS")
            history_tensors = []
            for padded_ids in padded_histories:
                vectors = [news2vector.get(nid, pad_vector) for nid in padded_ids]
                history_tensors.append(torch.stack(vectors))
            history_batch = torch.stack(history_tensors).to(device)

            if model_type == "lstur":  # For completeness; not used here.
                user_indices = torch.arange(len(chunk), device=device)
                user_vectors = model.get_user_vector(user_indices, history_batch)
            else:
                user_vectors = model.get_user_vector(history_batch)

            for sample_idx, sample in enumerate(chunk):
                candidate_vecs = []
                missing_candidate = False
                for news_id in sample["candidate_ids"]:
                    if news_id not in news2vector:
                        missing_candidate = True
                        break
                    candidate_vecs.append(news2vector[news_id])
                if missing_candidate:
                    # Skip impressions missing candidate vectors for this model.
                    continue

                candidate_tensor = torch.stack(candidate_vecs).unsqueeze(0).to(device)
                user_vec = user_vectors[sample_idx].unsqueeze(0)
                scores = model.get_prediction(candidate_tensor, user_vec).squeeze(0)
                scores_np = scores.detach().cpu().numpy()
                top_index = int(scores_np.argmax())

                results[sample["impression_id"]] = {
                    "scores": scores_np.tolist(),
                    "top_index": top_index,
                    "top_news_id": sample["candidate_ids"][top_index],
                    "top_label": int(sample["labels"][top_index]),
                }
    return results


def lexical_overlap(history_tokens: Iterable[str], candidate_tokens: Iterable[str]) -> float:
    history_set = set(history_tokens)
    candidate_set = set(candidate_tokens)
    if not candidate_set:
        return 0.0
    return len(history_set & candidate_set) / len(candidate_set)


def compute_click_rank(scores: Sequence[float], labels: Sequence[int]) -> int:
    if not scores:
        return 0
    sorted_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    for rank, (idx, _) in enumerate(sorted_pairs, start=1):
        if labels[idx] == 1:
            return rank
    return len(scores) + 1


def analyse_hits(
    samples: Sequence[dict],
    lsnr_preds: Dict[str, dict],
    nrms_preds: Dict[str, dict],
    news_tokens: Dict[str, List[str]],
) -> tuple[List[dict], Dict[str, dict]]:
    per_impression: List[dict] = []
    buckets = {
        "both_hit": defaultdict(list),
        "lsnr_only": defaultdict(list),
        "nrms_only": defaultdict(list),
        "both_fail": defaultdict(list),
    }

    for sample in samples:
        impression_id = sample["impression_id"]
        if impression_id not in lsnr_preds or impression_id not in nrms_preds:
            continue
        lsnr_pred = lsnr_preds[impression_id]
        nrms_pred = nrms_preds[impression_id]

        lsnr_hit = lsnr_pred["top_label"] == 1
        nrms_hit = nrms_pred["top_label"] == 1

        if lsnr_hit and nrms_hit:
            bucket = "both_hit"
        elif lsnr_hit and not nrms_hit:
            bucket = "lsnr_only"
        elif nrms_hit and not lsnr_hit:
            bucket = "nrms_only"
        else:
            bucket = "both_fail"

        history_tokens: List[str] = []
        for clicked_id in sample["clicked_news"]:
            history_tokens.extend(news_tokens.get(clicked_id, []))

        clicked_index = next((idx for idx, label in enumerate(sample["labels"]) if label == 1), None)
        if clicked_index is None:
            continue
        clicked_news_id = sample["candidate_ids"][clicked_index]
        clicked_tokens = news_tokens.get(clicked_news_id, [])
        clicked_overlap = lexical_overlap(history_tokens, clicked_tokens)

        negative_overlaps: List[float] = []
        for candidate_id, label in zip(sample["candidate_ids"], sample["labels"]):
            if label != 0:
                continue
            candidate_tokens = news_tokens.get(candidate_id, [])
            if not candidate_tokens:
                continue
            negative_overlaps.append(lexical_overlap(history_tokens, candidate_tokens))

        if negative_overlaps:
            best_negative_overlap = max(negative_overlaps)
            mean_negative_overlap = float(sum(negative_overlaps) / len(negative_overlaps))
        else:
            best_negative_overlap = 0.0
            mean_negative_overlap = 0.0

        lsnr_overlap = lexical_overlap(history_tokens, news_tokens.get(lsnr_pred["top_news_id"], []))
        nrms_overlap = lexical_overlap(history_tokens, news_tokens.get(nrms_pred["top_news_id"], []))

        lsnr_rank = compute_click_rank(lsnr_pred["scores"], sample["labels"])
        nrms_rank = compute_click_rank(nrms_pred["scores"], sample["labels"])

        lsnr_overlap_delta = lsnr_overlap - best_negative_overlap
        nrms_overlap_delta = nrms_overlap - best_negative_overlap
        overlap_delta = clicked_overlap - best_negative_overlap

        buckets[bucket]["count"].append(1)
        buckets[bucket]["lsnr_overlap"].append(lsnr_overlap)
        buckets[bucket]["nrms_overlap"].append(nrms_overlap)

        per_impression.append(
            {
                "impression_id": impression_id,
                "bucket": bucket,
                "clicked_news_count": len(sample["clicked_news"]),
                "candidate_count": len(sample["candidate_ids"]),
                "lsnr_top_news": lsnr_pred["top_news_id"],
                "nrms_top_news": nrms_pred["top_news_id"],
                "lsnr_hit": lsnr_hit,
                "nrms_hit": nrms_hit,
                "lsnr_overlap": lsnr_overlap,
                "nrms_overlap": nrms_overlap,
                "clicked_news_id": clicked_news_id,
                "clicked_overlap": clicked_overlap,
                "best_negative_overlap": best_negative_overlap,
                "mean_negative_overlap": mean_negative_overlap,
                "overlap_delta": overlap_delta,
                "lsnr_overlap_delta": lsnr_overlap_delta,
                "nrms_overlap_delta": nrms_overlap_delta,
                "lsnr_click_rank": lsnr_rank,
                "nrms_click_rank": nrms_rank,
            }
        )

    return per_impression, buckets


def summarise_buckets(buckets: Dict[str, dict]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for bucket_name, metrics in buckets.items():
        counts = metrics.get("count", [])
        total = len(counts)
        lsnr_avg = float(np.mean(metrics.get("lsnr_overlap", []))) if metrics.get("lsnr_overlap") else float("nan")
        nrms_avg = float(np.mean(metrics.get("nrms_overlap", []))) if metrics.get("nrms_overlap") else float("nan")
        summary[bucket_name] = {
            "count": total,
            "lsnr_overlap_mean": lsnr_avg,
            "nrms_overlap_mean": nrms_avg,
        }
    return summary


def print_overlap_hit_summary(per_impression: Sequence[dict]) -> None:
    if not per_impression:
        return

    df = pd.DataFrame(per_impression)
    if df.empty:
        return

    df["overlap_diff"] = df["lsnr_overlap"] - df["nrms_overlap"]
    tolerance = 1e-9

    def print_group(name: str, mask: pd.Series) -> None:
        subset = df[mask]
        if subset.empty:
            return
        total = int(subset.shape[0])
        lsnr_hits = int(subset["lsnr_hit"].sum())
        nrms_hits = int(subset["nrms_hit"].sum())
        mean_diff = float(subset["overlap_diff"].mean())
        print(
            f"{name:>24}: count={total:4d} | "
            f"LSNR hit={lsnr_hits:4d} ({lsnr_hits / total:.2%}) | "
            f"NRMS hit={nrms_hits:4d} ({nrms_hits / total:.2%}) | "
            f"mean Δ={mean_diff:.4f}"
        )

    print("\n=== Hit rate by overlap advantage ===")
    print_group("lsnr_overlap>nrms_overlap", df["overlap_diff"] > tolerance)
    print_group("nrms_overlap>lsnr_overlap", df["overlap_diff"] < -tolerance)
    print_group("overlap_tie", df["overlap_diff"].abs() <= tolerance)

    thresholds = [0.05, 0.10, 0.20]
    print("\n=== Hit rate when overlap advantage exceeds thresholds ===")
    for threshold in thresholds:
        label_pos = f"lsnr Δ≥{threshold:.2f}"
        label_neg = f"nrms Δ≥{threshold:.2f}"
        print_group(label_pos, df["overlap_diff"] >= threshold)
        print_group(label_neg, df["overlap_diff"] <= -threshold)


def print_click_rank_summary(per_impression: Sequence[dict]) -> None:
    if not per_impression:
        return

    df = pd.DataFrame(per_impression)
    required_columns = {"lsnr_click_rank", "nrms_click_rank"}
    if not required_columns.issubset(df.columns):
        return

    df = df.dropna(subset=["lsnr_click_rank", "nrms_click_rank"])
    if df.empty:
        return

    df["lsnr_click_rank"] = df["lsnr_click_rank"].astype(int)
    df["nrms_click_rank"] = df["nrms_click_rank"].astype(int)
    df["overlap_diff"] = df["lsnr_overlap"] - df["nrms_overlap"]

    lsnr_better = df[df["lsnr_click_rank"] < df["nrms_click_rank"]]
    nrms_better = df[df["nrms_click_rank"] < df["lsnr_click_rank"]]
    ties = df[df["lsnr_click_rank"] == df["nrms_click_rank"]]

    def describe(subset: pd.DataFrame, label: str) -> None:
        count = int(subset.shape[0])
        if count == 0:
            print(f"{label:>24}: count=   0")
            return
        mean_lsnr_rank = float(subset["lsnr_click_rank"].mean())
        mean_nrms_rank = float(subset["nrms_click_rank"].mean())
        mean_overlap_diff = float(subset["overlap_diff"].mean())
        print(
            f"{label:>24}: count={count:4d} | "
            f"mean rank (LSNR)={mean_lsnr_rank:.2f} | "
            f"mean rank (NRMS)={mean_nrms_rank:.2f} | "
            f"mean overlap Δ={mean_overlap_diff:.4f}"
        )

    print("\n=== Clicked document rank comparison ===")
    print(
        "Overall mean rank — LSNR: "
        f"{df['lsnr_click_rank'].mean():.2f}, NRMS: {df['nrms_click_rank'].mean():.2f}"
    )
    describe(lsnr_better, "lsnr rank < nrms rank")
    describe(nrms_better, "nrms rank < lsnr rank")
    describe(ties, "rank tie")


def _safe_mean(values: Iterable[float | None]) -> float:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return float("nan")
    return float(sum(filtered) / len(filtered))


def _format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def print_clicked_vs_negative_summary(per_impression: Sequence[dict], sample_count: int) -> None:
    rows = [row for row in per_impression if row.get("clicked_overlap") is not None]
    if not rows:
        return

    buckets: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get("bucket", "unknown")].append(row)

    print("\n=== Clicked vs negative lexical overlap ===")
    for bucket_name in ("both_hit", "lsnr_only", "nrms_only", "both_fail"):
        entries = buckets.get(bucket_name, [])
        if not entries:
            print(f"{bucket_name:>12}: count=0")
            continue

        count = len(entries)
        mean_clicked = _safe_mean(entry.get("clicked_overlap") for entry in entries)
        mean_best_negative = _safe_mean(entry.get("best_negative_overlap") for entry in entries)
        mean_delta = _safe_mean(entry.get("overlap_delta") for entry in entries)
        positive_delta = sum(1 for entry in entries if entry.get("overlap_delta", 0.0) > 0)
        lsnr_overlap = _safe_mean(entry.get("lsnr_overlap") for entry in entries)
        nrms_overlap = _safe_mean(entry.get("nrms_overlap") for entry in entries)
        lsnr_delta = _safe_mean(entry.get("lsnr_overlap_delta") for entry in entries)
        nrms_delta = _safe_mean(entry.get("nrms_overlap_delta") for entry in entries)

        frac_positive = positive_delta / count if count else float("nan")
        print(
            f"{bucket_name:>12}: count={count:4d} | "
            f"clicked={_format_float(mean_clicked)} | best_neg={_format_float(mean_best_negative)} | "
            f"Δ={_format_float(mean_delta)} | Δ>0={frac_positive:.2%} | "
            f"LSNR overlap={_format_float(lsnr_overlap)} | NRMS overlap={_format_float(nrms_overlap)} | "
            f"LSNR Δ={_format_float(lsnr_delta)} | NRMS Δ={_format_float(nrms_delta)}"
        )

    overall_clicked = _safe_mean(row.get("clicked_overlap") for row in rows)
    overall_best_negative = _safe_mean(row.get("best_negative_overlap") for row in rows)
    overall_delta = _safe_mean(row.get("overlap_delta") for row in rows)
    print(
        "\nOverall: "
        f"count={len(rows)} | clicked={_format_float(overall_clicked)} | "
        f"best_neg={_format_float(overall_best_negative)} | Δ={_format_float(overall_delta)}"
    )

    if sample_count > 0:
        for label in ("lsnr_only", "nrms_only"):
            entries = sorted(buckets.get(label, []), key=lambda item: item.get("overlap_delta", 0.0))
            if not entries:
                continue
            print(
                f"\nTop {min(sample_count, len(entries))} samples for {label} "
                "(sorted by overlap delta ascending):"
            )
            for entry in entries[:sample_count]:
                print(
                    f"  impression={entry['impression_id']} | history_len={entry['clicked_news_count']} | "
                    f"clicked={_format_float(entry.get('clicked_overlap'))} | "
                    f"best_neg={_format_float(entry.get('best_negative_overlap'))} | "
                    f"Δ={_format_float(entry.get('overlap_delta'))} | "
                    f"LSNR Δ={_format_float(entry.get('lsnr_overlap_delta'))} | "
                    f"NRMS Δ={_format_float(entry.get('nrms_overlap_delta'))}"
                )

def main() -> None:
    args = parse_args()

    lsnr_model, lsnr_device, lsnr_config = load_model(
        args.lsnr_experiment, args.dataset, args.lsnr_checkpoint_dir
    )
    nrms_model, nrms_device, nrms_config = load_model(
        args.nrms_experiment, args.dataset, args.nrms_checkpoint_dir
    )

    if lsnr_device.type != nrms_device.type:
        print(
            f"Warning: LSNR device is {lsnr_device.type} while NRMS device is {nrms_device.type}. "
            "Proceeding, but performance may differ.",
            flush=True,
        )

    dataset_path_lsnr = Path(lsnr_config["datasets"][args.dataset]["path"])
    dataset_path_nrms = Path(nrms_config["datasets"][args.dataset]["path"])
    if dataset_path_lsnr != dataset_path_nrms:
        raise ValueError(
            "LSNR and NRMS configs reference different dataset paths. "
            f"Got {dataset_path_lsnr} vs {dataset_path_nrms}."
        )
    dataset_path = dataset_path_lsnr

    cold_thresholds = list(dict.fromkeys(args.cold_threshold))  # preserve order, drop duplicates

    def resolve_news_device(requested: str | None, fallback: torch.device) -> torch.device:
        if requested is None:
            return fallback
        try:
            candidate = get_device(requested)
        except Exception as exc:  # pragma: no cover - defensive logging path
            print(
                f"Warning: unable to use requested news device '{requested}': {exc}. "
                f"Falling back to {fallback}."
            )
            return fallback

        if candidate.type == "mps":
            try:
                torch.zeros(1, device=candidate)
            except RuntimeError as exc:
                print(
                    "Warning: MPS device requested for news encoding but is unavailable ("
                    f"{exc}). Falling back to {fallback}."
                )
                return fallback
        return candidate

    requested_news_device = args.news_device
    lsnr_news_device = resolve_news_device(requested_news_device, lsnr_device)
    nrms_news_device = resolve_news_device(requested_news_device, nrms_device)

    behavior_files = []
    for cold_threshold in cold_thresholds:
        behaviors_file = dataset_path / args.split / f"behaviors_parsed_cold{cold_threshold}.tsv"
        if not behaviors_file.exists():
            raise FileNotFoundError(f"Behaviors file not found: {behaviors_file}")
        behavior_files.append((cold_threshold, behaviors_file))

    # Build news vectors per model once, reused for all thresholds.
    print("Building LSNR news vectors...")
    original_lsnr_device = lsnr_device
    if lsnr_news_device != lsnr_device:
        try:
            lsnr_model.to(lsnr_news_device)
            print(f"LSNR model moved to {lsnr_news_device} for news encoding")
        except RuntimeError as exc:
            print(
                f"Warning: failed to move LSNR model to {lsnr_news_device} ({exc}). "
                f"Using {lsnr_device} instead."
            )
            lsnr_news_device = lsnr_device

    lsnr_news2vector, _ = build_news_vectors(
        lsnr_model,
        lsnr_config,
        args.dataset,
        lsnr_config["experiment"]["model"],
        lsnr_config["training"]["batch_size"],
        lsnr_config["training"].get("num_workers", 4),
        cold_threshold=0,
        split=args.split,
        device=lsnr_news_device,
    )

    if lsnr_news_device != original_lsnr_device:
        lsnr_model.to(original_lsnr_device)
        print(f"LSNR model returned to {original_lsnr_device} for scoring")

    print("Building NRMS news vectors...")
    original_nrms_device = nrms_device
    if nrms_news_device != nrms_device:
        try:
            nrms_model.to(nrms_news_device)
            print(f"NRMS model moved to {nrms_news_device} for news encoding")
        except RuntimeError as exc:
            print(
                f"Warning: failed to move NRMS model to {nrms_news_device} ({exc}). "
                f"Using {nrms_device} instead."
            )
            nrms_news_device = nrms_device

    nrms_news2vector, _ = build_news_vectors(
        nrms_model,
        nrms_config,
        args.dataset,
        nrms_config["experiment"]["model"],
        nrms_config["training"]["batch_size"],
        nrms_config["training"].get("num_workers", 4),
        cold_threshold=0,
        split=args.split,
        device=nrms_news_device,
    )

    if nrms_news_device != original_nrms_device:
        nrms_model.to(original_nrms_device)
        print(f"NRMS model returned to {original_nrms_device} for scoring")

    # Build lexical token map (use train/val/test to cover history/news)
    print("Building lexical token map...")
    news_tokens = build_news_token_map(dataset_path, ("train", "val", "test"))

    def make_output_path(base: str | None, cold_threshold: int) -> Path | None:
        if not base:
            return None
        output_path = Path(base)
        if output_path.is_dir() or not output_path.suffix:
            output_path = output_path if output_path.is_dir() else output_path
            return output_path / f"lsnr_nrms_overlap_cold{cold_threshold}.csv"
        stem = output_path.stem
        return output_path.with_name(f"{stem}_cold{cold_threshold}{output_path.suffix}")

    for cold_threshold, behaviors_file in behavior_files:
        print(f"\n===== Cold threshold {cold_threshold} =====")
        print(f"Loading behaviors from {behaviors_file}")
        samples = load_behaviors(behaviors_file, args.max_impressions)
        if not samples:
            print("No valid impressions found after filtering; skipping threshold.")
            continue
        print(f"Loaded {len(samples)} impressions for analysis")

        print("Collecting LSNR predictions...")
        lsnr_predictions = collect_predictions(
            lsnr_model,
            lsnr_config["experiment"]["model"],
            lsnr_device,
            samples,
            lsnr_news2vector,
            args.batch_size,
        )
        print(f"LSNR predictions computed for {len(lsnr_predictions)} impressions")

        print("Collecting NRMS predictions...")
        nrms_predictions = collect_predictions(
            nrms_model,
            nrms_config["experiment"]["model"],
            nrms_device,
            samples,
            nrms_news2vector,
            args.batch_size,
        )
        print(f"NRMS predictions computed for {len(nrms_predictions)} impressions")

        per_impression, buckets = analyse_hits(samples, lsnr_predictions, nrms_predictions, news_tokens)
        if not per_impression:
            print("No overlapping impressions between LSNR and NRMS results; skipping threshold.")
            continue

        summary = summarise_buckets(buckets)

        total_considered = sum(bucket_info["count"] for bucket_info in summary.values())
        if total_considered == 0:
            print("No impressions remained after aligning model predictions; skipping threshold.")
            continue
        lsnr_hits = summary.get("both_hit", {}).get("count", 0) + summary.get("lsnr_only", {}).get("count", 0)
        nrms_hits = summary.get("both_hit", {}).get("count", 0) + summary.get("nrms_only", {}).get("count", 0)

        print("\n=== Hit/Fail Comparison ===")
        print(f"Total impressions analysed: {total_considered}")
        print(f"LSNR hits: {lsnr_hits} ({lsnr_hits / total_considered:.2%})")
        print(f"NRMS hits: {nrms_hits} ({nrms_hits / total_considered:.2%})")

        print("\n=== Lexical Overlap by Outcome ===")
        for bucket_name in ["both_hit", "lsnr_only", "nrms_only", "both_fail"]:
            bucket_info = summary.get(bucket_name, {"count": 0})
            print(
                f"{bucket_name:>10}: count={bucket_info.get('count', 0):4d} | "
                f"LSNR overlap={bucket_info.get('lsnr_overlap_mean', float('nan')):.4f} | "
                f"NRMS overlap={bucket_info.get('nrms_overlap_mean', float('nan')):.4f}"
            )

        print_overlap_hit_summary(per_impression)
        print_click_rank_summary(per_impression)
        print_clicked_vs_negative_summary(per_impression, args.overlap_sample_count)

        output_path = make_output_path(args.output, cold_threshold)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(per_impression).to_csv(output_path, index=False)
            print(f"Saved per-impression analysis to {output_path}")


if __name__ == "__main__":
    main()
