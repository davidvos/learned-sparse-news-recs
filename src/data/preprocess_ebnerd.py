"""EBNeRD preprocessing.

This script transforms the EBNeRD parquet data dumps into the MIND-style
artifacts consumed by the training and evaluation pipelines. The conversion
pipeline performs the following steps:

* load dataset configuration from `config.yaml`
* materialise user browsing histories from the list-style parquet schema
* derive MIND-compatible train/validation/test behavior files
* export news metadata shards restricted to the articles that actually appear
  in the processed splits
* build the `user2int` and `category2int` lookup tables shared by all splits

The resulting directory structure under `datasets.ebnerd.path` matches the
layout expected by the other scripts in this repository.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import load_config


def _unique_nonempty_str(values: Iterable[object]) -> List[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values or []:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):  # type: ignore[arg-type]
            continue
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _load_articles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing articles parquet at {path}")
    df = pd.read_parquet(path)

    def first_subcategory(entry: object) -> str:
        if isinstance(entry, (list, tuple, np.ndarray)) and len(entry) > 0:
            return str(entry[0])
        return ""

    category = df["category_str"].fillna("").astype(str) if "category_str" in df.columns else df["category"].astype(str)
    subcategory_series = df["subcategory"] if "subcategory" in df.columns else pd.Series(["" for _ in range(len(df))])

    news = pd.DataFrame(
        {
            "id": df["article_id"].astype(str),
            "category": category,
            "subcategory": subcategory_series.apply(first_subcategory),
            "title": df["title"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip() if "title" in df.columns else "",
            "abstract": df["subtitle"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip() if "subtitle" in df.columns else "",
            "body": df["body"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip() if "body" in df.columns else "",
            "url": df["url"].fillna("").astype(str) if "url" in df.columns else "",
            "title_entities": "",
            "abstract_entities": "",
        }
    )
    news = news.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    return news


def _explode_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing history parquet at {path}")
    df = pd.read_parquet(path)
    list_columns = [
        col
        for col in ("impression_time_fixed", "scroll_percentage_fixed", "article_id_fixed", "read_time_fixed")
        if col in df.columns
    ]
    if not list_columns:
        raise ValueError(f"History parquet at {path} does not contain expected list columns")
    exploded = df.explode(list_columns)
    exploded = exploded.dropna(subset=["impression_time_fixed", "article_id_fixed"])
    exploded = exploded.rename(
        columns={
            "user_id": "user",
            "impression_time_fixed": "time",
            "article_id_fixed": "article_id",
        }
    )
    exploded["user"] = exploded["user"].astype(str)
    exploded["time"] = pd.to_datetime(exploded["time"])
    exploded["article_id"] = exploded["article_id"].astype(str)
    exploded = exploded.sort_values(["user", "time"]).reset_index(drop=True)
    return exploded[["user", "time", "article_id"]]


def _build_history_lookup(events: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for user, group in events.groupby("user"):
        times = group["time"].to_numpy(dtype="datetime64[ns]")
        articles = group["article_id"].to_numpy(dtype=str)
        lookup[user] = (times, articles)
    return lookup


def _history_for(lookup: Dict[str, Tuple[np.ndarray, np.ndarray]], user: str, cutoff: pd.Timestamp, max_len: int) -> str:
    if pd.isna(cutoff):
        return ""
    data = lookup.get(user)
    if data is None:
        return ""
    times, articles = data
    idx = np.searchsorted(times, np.array([cutoff], dtype="datetime64[ns]")[0], side="left")
    if idx == 0:
        return ""
    history_slice = articles[max(0, idx - max_len):idx] if max_len and max_len > 0 else articles[:idx]
    return " ".join(history_slice.tolist())


def _prepare_behaviors(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing behaviors parquet at {path}")
    df = pd.read_parquet(path)
    df = df.rename(
        columns={
            "user_id": "user",
            "impression_time": "time",
        }
    )
    df["user"] = df["user"].astype(str)
    df["time"] = pd.to_datetime(df["time"])

    def to_list(obj: object) -> list[object]:
        if obj is None:
            return []
        if isinstance(obj, list):
            return obj
        if isinstance(obj, tuple):
            return list(obj)
        if hasattr(obj, "to_pylist"):
            return list(obj.to_pylist())  # type: ignore[attr-defined]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return [obj]

    df["inview_list"] = df.get("article_ids_inview", pd.Series([], dtype=object)).apply(to_list)
    df["clicked_list"] = df.get("article_ids_clicked", pd.Series([], dtype=object)).apply(to_list)
    df["inview_list"] = df["inview_list"].apply(_unique_nonempty_str)
    df["clicked_list"] = df["clicked_list"].apply(_unique_nonempty_str)
    return df[["impression_id", "user", "time", "inview_list", "clicked_list"]]


def _format_impressions(inview: Sequence[str], clicked: Sequence[str]) -> str:
    clicked_set = set(clicked)
    ordered: list[str] = []
    for nid in inview:
        label = 1 if nid in clicked_set else 0
        ordered.append(f"{nid}-{label}")
    for nid in clicked:
        if nid not in inview:
            ordered.append(f"{nid}-1")
    return " ".join(ordered)


def _build_eval_behaviors(
    behaviors: pd.DataFrame,
    history_lookup: Dict[str, Tuple[np.ndarray, np.ndarray]],
    history_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in behaviors.itertuples(index=False):
        history = _history_for(history_lookup, row.user, row.time, history_size)
        impressions = _format_impressions(row.inview_list, row.clicked_list)
        rows.append(
            {
                "impression_id": str(getattr(row, "impression_id")),
                "user": row.user,
                "time": row.time,
                "clicked_news": history,
                "impressions": impressions,
                "candidate_news": " ".join([token.split("-")[0] for token in impressions.split()]),
                "clicked": " ".join([token.split("-")[1] for token in impressions.split()]),
            }
        )
    result = pd.DataFrame(rows)
    result["time"] = result["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    result["clicked_news"] = result["clicked_news"].fillna("")
    result["impressions"] = result["impressions"].fillna("")
    return result


def _build_train_behaviors(
    behaviors: pd.DataFrame,
    history_lookup: Dict[str, Tuple[np.ndarray, np.ndarray]],
    history_size: int,
) -> pd.DataFrame:
    """Prepare training behaviors with all available negatives from the impression."""
    rows: list[dict[str, object]] = []
    for row in behaviors.itertuples(index=False):
        positives = _unique_nonempty_str(row.clicked_list)
        if not positives:
            continue
        negatives = [nid for nid in row.inview_list if nid not in positives]
        if not negatives:
            continue

        history = _history_for(history_lookup, row.user, row.time, history_size)
        for pos_id in positives:
            candidates = [pos_id] + negatives
            labels = ["1"] + ["0"] * len(negatives)
            rows.append(
                {
                    "user": row.user,
                    "clicked_news": history,
                    "candidate_news": " ".join(candidates),
                    "clicked": " ".join(labels),
                }
            )
    result = pd.DataFrame(rows)
    for col in ("clicked_news", "candidate_news", "clicked"):
        if col in result.columns:
            result[col] = result[col].fillna("")
    return result


def _collect_ids_from_space_strings(series: pd.Series) -> set[str]:
    ids: set[str] = set()
    for value in series.dropna().astype(str):
        if not value.strip():
            continue
        ids.update(part for part in value.split() if part)
    return ids


def _collect_ids_from_impressions(series: pd.Series) -> set[str]:
    ids: set[str] = set()
    for value in series.dropna().astype(str):
        if not value.strip():
            continue
        for token in value.split():
            if "-" in token:
                nid, _ = token.rsplit("-", 1)
                if nid:
                    ids.add(nid)
    return ids


def _write_news_split(news_df: pd.DataFrame, ids: set[str], out_path: Path) -> None:
    keep = news_df[news_df["id"].isin(ids)].drop_duplicates(subset=["id"], keep="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep[[
        "id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "body",
        "url",
        "title_entities",
        "abstract_entities",
    ]].to_csv(out_path, sep="\t", index=False, header=False)


def _create_mappings(out_dir: Path, behaviors: Sequence[pd.DataFrame], news_df: pd.DataFrame) -> None:
    user_series = [df["user"].astype(str) for df in behaviors if "user" in df.columns]
    if not user_series:
        raise ValueError("No user columns provided to build user2int mapping")
    users = pd.Series(pd.concat(user_series, ignore_index=True).unique()).sort_values()
    
    categories = pd.Series(news_df["category"].dropna().astype(str).unique()).sort_values()
    subcategories = pd.Series(news_df["subcategory"].dropna().astype(str).unique()).sort_values()
    # Filter out empty strings if any
    categories = categories[categories.str.strip() != ""]
    subcategories = subcategories[subcategories.str.strip() != ""]

    user2int = pd.DataFrame({"user": users.tolist(), "int": range(1, len(users) + 1)})
    cat2int = pd.DataFrame({"category": categories.tolist(), "int": range(1, len(categories) + 1)})
    subcat2int = pd.DataFrame({"subcategory": subcategories.tolist(), "int": range(1, len(subcategories) + 1)})

    user2int.to_csv(out_dir / "user2int.tsv", sep="\t", index=False, header=False)
    cat2int.to_csv(out_dir / "category2int.tsv", sep="\t", index=False, header=False)
    subcat2int.to_csv(out_dir / "subcategory2int.tsv", sep="\t", index=False, header=False)




def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(os.environ.get("PREPROCESS_CONFIG", project_root / "config.yaml"))
    config = load_config(str(cfg_path))

    dataset_cfg = config.get("datasets", {}).get("ebnerd")
    if dataset_cfg is None:
        raise ValueError("Add 'ebnerd' dataset configuration to config.yaml under datasets.ebnerd")

    base_dir = Path(dataset_cfg["path"]).resolve()
    articles_path = base_dir / "articles.parquet"
    train_beh_path = base_dir / "train" / "behaviors.parquet"
    train_hist_path = base_dir / "train" / "history.parquet"
    val_beh_path = base_dir / "validation" / "behaviors.parquet"
    val_hist_path = base_dir / "validation" / "history.parquet"

    news_df = _load_articles(articles_path)
    train_history_events = _explode_history(train_hist_path)
    val_history_events = _explode_history(val_hist_path)
    train_history_lookup = _build_history_lookup(train_history_events)
    val_history_lookup = _build_history_lookup(val_history_events)
    train_behaviors = _prepare_behaviors(train_beh_path)
    official_val_behaviors = _prepare_behaviors(val_beh_path)

    history_size = int(dataset_cfg.get("num_clicked_news_a_user", 30))
    out_train = base_dir / "train"
    out_val = base_dir / "val"
    out_test = base_dir / "test"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    # --- 90/10 Chronological Split ---
    # Sort training behaviors by time, take last 10% as validation
    train_behaviors = train_behaviors.sort_values("time").reset_index(drop=True)
    total_rows = len(train_behaviors)
    split_index = int(total_rows * 0.9)
    split_index = min(max(split_index, 1), total_rows - 1)
    
    print(f"Splitting EBNeRD train behaviors — train rows: {split_index}, validation rows (10%): {total_rows - split_index}")
    
    train_split = train_behaviors.iloc[:split_index].copy()
    val_split = train_behaviors.iloc[split_index:].copy()
    
    # Official validation becomes test set
    test_behaviors = official_val_behaviors.copy()

    print("Generating training data")
    train_parsed = _build_train_behaviors(train_split, train_history_lookup, history_size)
    train_parsed[["user", "clicked_news", "candidate_news", "clicked"]].to_csv(
        out_train / "behaviors_parsed.tsv",
        sep="\t",
        index=False,
        header=True,
    )

    # Validation set (10% of train)
    val_eval = _build_eval_behaviors(val_split, train_history_lookup, history_size)
    
    # Test set (official validation)
    test_eval = _build_eval_behaviors(test_behaviors, val_history_lookup, history_size)

    val_eval[["impression_id", "user", "time", "clicked_news", "impressions"]].to_csv(
        out_val / "behaviors.tsv",
        sep="\t",
        index=False,
        header=False,
    )
    test_eval[["impression_id", "user", "time", "clicked_news", "impressions"]].to_csv(
        out_test / "behaviors.tsv",
        sep="\t",
        index=False,
        header=False,
    )

    train_ids = _collect_ids_from_space_strings(train_parsed["candidate_news"]) | _collect_ids_from_space_strings(train_parsed["clicked_news"])
    val_ids = (
        _collect_ids_from_impressions(val_eval["impressions"]) |
        _collect_ids_from_space_strings(val_eval["clicked_news"])
    )
    test_ids = (
        _collect_ids_from_impressions(test_eval["impressions"]) |
        _collect_ids_from_space_strings(test_eval["clicked_news"])
    )
    val_union_ids = train_ids | val_ids
    test_union_ids = train_ids | val_ids | test_ids

    _write_news_split(news_df, train_ids, out_train / "news.tsv")
    _write_news_split(news_df, val_union_ids, out_val / "news.tsv")
    _write_news_split(news_df, test_union_ids, out_test / "news.tsv")

    _create_mappings(out_train, [train_parsed, val_eval, test_eval], news_df)

    val_cols = ["impression_id", "user", "time", "clicked_news", "impressions", "candidate_news", "clicked"]
    val_eval[val_cols].to_csv(out_val / "behaviors_parsed.tsv", sep="\t", index=False, header=True)
    test_eval[val_cols].to_csv(out_test / "behaviors_parsed.tsv", sep="\t", index=False, header=True)

    print("EBNeRD preprocessing complete.")


if __name__ == "__main__":
    main()


