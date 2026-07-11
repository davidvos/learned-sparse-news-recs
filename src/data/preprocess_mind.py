"""Preprocessing script for the MIND news recommendation dataset.

Constructs validation and test splits by holding out the latest periods
from the original training behaviors file, following the chronological
evaluation protocol used in recent recommendation literature.
Outputs are saved as tab-separated files ready for model training and
evaluation.

Usage::

    uv run python src/data/preprocess_mind.py
"""
from __future__ import annotations

import os
from os import path
import random
from pathlib import Path
from typing import List, Sequence, Set

import pandas as pd
from tqdm import tqdm

from config import load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
CONFIG_PATH = Path(os.environ.get("PREPROCESS_CONFIG", DEFAULT_CONFIG_PATH))
config = load_config(str(CONFIG_PATH))

# Extract relevant config sections for easier access
global_config = config["global"]
training_config = config["training"]
datasets_config = config["datasets"]


def _normalize_clicked_news(text: str) -> str:
    """Normalize whitespace in clicked history strings."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def parse_behaviors_df(behaviors: pd.DataFrame) -> pd.DataFrame:
    """Prepare training behaviors by pairing each positive with all negatives from the same impression.
    
    Actual negative sampling will happen dynamically during training in the Dataset class.

    Args:
        behaviors: DataFrame with columns impression_id, user, time,
            clicked_news, impressions (MIND schema).

    Returns:
        DataFrame with columns user, clicked_news, candidate_news, clicked.
    """
    if behaviors.empty:
        return pd.DataFrame(columns=["user", "clicked_news", "candidate_news", "clicked"])

    records: List[dict] = []

    tmp = behaviors.copy()
    tmp["clicked_news"] = tmp["clicked_news"].fillna(" ").map(_normalize_clicked_news)
    tmp["impressions_list"] = tmp["impressions"].str.split()

    for row in tqdm(tmp.itertuples(index=False), total=len(tmp), desc="Preparing train impressions"):
        impressions_list = row.impressions_list
        if not impressions_list:
            continue

        positives = [candidate for candidate in impressions_list if candidate.endswith("1")]
        negatives = [candidate for candidate in impressions_list if candidate.endswith("0")]

        if not positives or not negatives:
            continue

        neg_ids = [item.split("-")[0] for item in negatives]
        neg_labels = ["0"] * len(neg_ids)

        for pos in positives:
            pos_id = pos.split("-")[0]
            candidate_news = " ".join([pos_id] + neg_ids)
            clicked = " ".join(["1"] + neg_labels)
            records.append(
                {
                    "user": row.user,
                    "clicked_news": row.clicked_news,
                    "candidate_news": candidate_news,
                    "clicked": clicked,
                }
            )

    return pd.DataFrame(records, columns=["user", "clicked_news", "candidate_news", "clicked"])


def parse_eval_behaviors_df(
    behaviors: pd.DataFrame, min_history: int,
) -> pd.DataFrame:
    """Convert evaluation behaviors into MIND-style impression rows.

    Args:
        behaviors: DataFrame with columns impression_id, user, time,
            clicked_news, impressions (MIND schema).
        min_history: Minimum history length required to keep an impression.

    Returns:
        DataFrame with columns impression_id, user, time, clicked_news,
        impressions, candidate_news, clicked suitable for evaluation.
    """
    if behaviors.empty:
        return pd.DataFrame(
            columns=[
                "impression_id",
                "user",
                "time",
                "clicked_news",
                "impressions",
                "candidate_news",
                "clicked",
            ]
        )

    tmp = behaviors.copy()
    tmp["clicked_news"] = tmp["clicked_news"].fillna(" ").map(_normalize_clicked_news)


    if min_history > 0:
        tmp = tmp[tmp["clicked_news"].apply(lambda x: len(x.split())) >= min_history]

    tmp["impressions_list"] = tmp["impressions"].str.split()
    tmp = tmp[tmp["impressions_list"].map(bool)]

    def _parse_impression_tokens(tokens: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Return article ids, click labels, and cleaned tokens, skipping malformed entries."""
        articles: list[str] = []
        labels: list[str] = []
        cleaned: list[str] = []
        for token in tokens:
            if not isinstance(token, str):
                continue
            article, sep, label = token.rpartition("-")
            if sep == "" or not article:
                continue
            articles.append(article)
            labels.append(label)
            cleaned.append(f"{article}-{label}")
        return articles, labels, cleaned

    parsed_tokens = tmp["impressions_list"].apply(_parse_impression_tokens)
    parsed_df = parsed_tokens.apply(pd.Series)
    parsed_df.columns = ["candidate_news_list", "clicked_list", "cleaned_tokens"]

    tmp["candidate_news"] = parsed_df["candidate_news_list"].apply(lambda items: " ".join(items))
    tmp["clicked"] = parsed_df["clicked_list"].apply(lambda items: " ".join(items))
    tmp["impressions"] = parsed_df["cleaned_tokens"].apply(lambda items: " ".join(items))

    tmp = tmp[tmp["candidate_news"].str.strip().astype(bool)]


    return tmp[[
        "impression_id",
        "user",
        "time",
        "clicked_news",
        "impressions",
        "candidate_news",
        "clicked",
    ]]


def collect_article_ids(behaviors: pd.DataFrame) -> Set[str]:
    """Collect all article IDs referenced in behaviors rows."""
    article_ids: Set[str] = set()
    if behaviors.empty:
        return article_ids

    clicked_series = behaviors["clicked_news"].fillna(" ").astype(str)
    for history in clicked_series:
        if not history:
            continue
        article_ids.update(piece for piece in history.split() if piece)

    impressions_series = behaviors["impressions"].fillna("")
    for impression in impressions_series:
        for item in str(impression).split():
            if "-" in item:
                article_ids.add(item.split("-")[0])
            else:
                article_ids.add(item)

    return article_ids


def write_news_split(news_df: pd.DataFrame, article_ids: Set[str], output_path: str) -> None:
    """Write a split-specific news file, falling back to full set if empty."""
    if article_ids:
        subset = news_df[news_df["id"].isin(article_ids)].copy()
        if subset.empty:
            subset = news_df.copy()
    else:
        subset = news_df.copy()

    subset.to_csv(output_path, sep="\t", index=False, header=False)


def create_user2int_category2int(
    behaviors_dfs: Sequence[pd.DataFrame],
    news_dfs: Sequence[pd.DataFrame],
    output_dirs: Sequence[str],
) -> None:
    """Create shared mapping files across all provided splits."""
    all_users: Set[str] = set()
    for df in behaviors_dfs:
        if df is None or df.empty:
            continue
        all_users.update(df["user"].astype(str).unique().tolist())

    user2int = {user: idx + 1 for idx, user in enumerate(sorted(all_users))}
    user2int_df = pd.DataFrame(list(user2int.items()), columns=["user", "int"])

    all_categories: Set[str] = set()
    all_subcategories: Set[str] = set()
    for df in news_dfs:
        if df is None or df.empty:
            continue
        all_categories.update(df["category"].dropna().astype(str).tolist())
        all_subcategories.update(df["subcategory"].dropna().astype(str).tolist())

    category2int = {cat: idx + 1 for idx, cat in enumerate(sorted(all_categories))}
    category2int_df = pd.DataFrame(list(category2int.items()), columns=["category", "int"])

    subcategory2int = {subcat: idx + 1 for idx, subcat in enumerate(sorted(all_subcategories))}
    subcategory2int_df = pd.DataFrame(list(subcategory2int.items()), columns=["subcategory", "int"])

    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
        user2int_df.to_csv(path.join(directory, "user2int.tsv"), sep="\t", index=False, header=False)
        category2int_df.to_csv(path.join(directory, "category2int.tsv"), sep="\t", index=False, header=False)
        subcategory2int_df.to_csv(path.join(directory, "subcategory2int.tsv"), sep="\t", index=False, header=False)


def _read_behaviors(path_to_file: str) -> pd.DataFrame:
    columns = ["impression_id", "user", "time", "clicked_news", "impressions"]
    return pd.read_table(path_to_file, header=None, names=columns)


def _read_news(path_to_file: str) -> pd.DataFrame:
    columns = [
        "id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    return pd.read_table(path_to_file, header=None, names=columns)


if __name__ == "__main__":
    base_path = datasets_config["mind"]["path"]
    
    # Source directories (raw MIND data)
    source_train_dir = path.join(base_path, "MINDsmall_train")
    source_dev_dir = path.join(base_path, "MINDsmall_dev")
    
    # Output directories (processed data)
    train_dir = path.join(base_path, "train")
    val_dir = path.join(base_path, "val")
    test_dir = path.join(base_path, "test")

    for directory in (train_dir, val_dir, test_dir):
        os.makedirs(directory, exist_ok=True)

    # Read from source directories
    behaviors_path = path.join(source_train_dir, "behaviors.tsv")
    news_path = path.join(source_train_dir, "news.tsv")

    if not path.exists(behaviors_path):
        raise FileNotFoundError(f"Missing training behaviors file: {behaviors_path}")
    if not path.exists(news_path):
        raise FileNotFoundError(f"Missing training news file: {news_path}")

    behaviors = _read_behaviors(behaviors_path)
    train_news_df = _read_news(news_path)

    # Read official dev set (will become test set)
    original_dev_behaviors_path = path.join(source_dev_dir, "behaviors.tsv")
    original_dev_news_path = path.join(source_dev_dir, "news.tsv")
    if not path.exists(original_dev_behaviors_path) or not path.exists(original_dev_news_path):
        raise FileNotFoundError(
            "Expected original MIND dev files (behaviors.tsv and news.tsv) to exist under "
            f"{source_dev_dir}."
        )
    official_dev_behaviors = _read_behaviors(original_dev_behaviors_path)
    official_dev_news_df = _read_news(original_dev_news_path)

    # Chronological split parameters
    mind_cfg = datasets_config["mind"]
    min_history_eval = int(mind_cfg.get("min_history_for_eval", 1))

    behaviors["timestamp"] = pd.to_datetime(behaviors["time"], errors="coerce")
    behaviors = behaviors.dropna(subset=["timestamp"]).copy()
    behaviors.sort_values("timestamp", inplace=True)

    total_rows = len(behaviors)
    if total_rows < 10:
        raise ValueError("Expected at least 10 impressions to perform a 90/10 split.")

    split_index = int(total_rows * 0.9)
    split_index = min(max(split_index, 1), total_rows - 1)

    print(
        "Splitting MIND train behaviors — train rows: {}, validation rows (10%): {}".format(
            split_index, total_rows - split_index
        )
    )

    train_behaviors = behaviors.iloc[:split_index].copy()
    val_behaviors = behaviors.iloc[split_index:].copy()
    test_behaviors = official_dev_behaviors.copy()

    for df in (train_behaviors, val_behaviors, test_behaviors):
        df.drop(columns=["timestamp"], inplace=True, errors="ignore")

    # Persist raw evaluation splits to ease inspection / reuse
    if not val_behaviors.empty:
        val_behaviors.to_csv(path.join(val_dir, "behaviors.tsv"), sep="\t", index=False, header=False)
    if not test_behaviors.empty:
        test_behaviors.to_csv(path.join(test_dir, "behaviors.tsv"), sep="\t", index=False, header=False)

    # Prepare news splits (include union of articles referenced in each split)
    combined_news_df = (
        pd.concat([train_news_df, official_dev_news_df], ignore_index=True)
        .drop_duplicates(subset=["id"], keep="last")
    )

    train_news_ids = collect_article_ids(train_behaviors)
    val_news_ids = collect_article_ids(pd.concat([train_behaviors, val_behaviors], ignore_index=True))
    test_news_ids = collect_article_ids(pd.concat([train_behaviors, val_behaviors, test_behaviors], ignore_index=True))

    write_news_split(train_news_df, train_news_ids, path.join(train_dir, "news.tsv"))
    write_news_split(combined_news_df, val_news_ids, path.join(val_dir, "news.tsv"))
    write_news_split(combined_news_df, test_news_ids, path.join(test_dir, "news.tsv"))

    # Create mapping files shared across splits
    create_user2int_category2int(
        [train_behaviors, val_behaviors, test_behaviors],
        [train_news_df, combined_news_df, combined_news_df],
        [train_dir, val_dir, test_dir],
    )

    print("Process data for training")
    print("Parse behaviors")
    train_output_path = path.join(train_dir, "behaviors_parsed.tsv")
    train_processed = parse_behaviors_df(train_behaviors)
    train_processed.to_csv(train_output_path, sep="\t", index=False)

    print("Process data for validation")
    val_processed = parse_eval_behaviors_df(val_behaviors, min_history=0)
    val_processed.to_csv(path.join(val_dir, "behaviors_parsed.tsv"), sep="\t", index=False, header=True)

    print("Process data for test")
    test_processed = parse_eval_behaviors_df(test_behaviors, min_history=0)
    test_processed.to_csv(path.join(test_dir, "behaviors_parsed.tsv"), sep="\t", index=False, header=True)

    print("Preprocessing complete.")
