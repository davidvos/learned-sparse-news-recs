"""Pre-tokenization script.

Script to pre-tokenize news content and save as a PyTorch tensor dictionary.
This avoids on-the-fly tokenization bottlenecks during training by producing
a single `news_tokenized.pt` cache artifact.

Usage:
    python src/data/tokenize.py --dataset mind
    python src/data/tokenize.py --dataset ebnerd --output data/ebnerd/news_tokenized.pt
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import load_config
from src.data.dataset import _load_news_df, _load_mapping


def tokenize_news(dataset_name: str, output_path: str | None = None) -> None:
    """
    Tokenize news articles for a specific dataset and save them as a PyTorch dictionary.

    Args:
        dataset_name: Name of the dataset defined in config.yaml (e.g., 'mind').
        output_path: Optional path to save the tokenized output. Defaults to <dataset_path>/news_tokenized.pt.
    """
    config_path = ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    config = load_config(str(config_path))
    
    if dataset_name not in config["datasets"]:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")

    dataset_cfg = config["datasets"][dataset_name]
    base_path = Path(dataset_cfg["path"])
    print(f"Processing dataset: {dataset_name} at {base_path}")

    # Check for splits
    splits = ["train", "val"]
    test_path = base_path / "test"
    if test_path.exists():
        splits.append("test")

    dfs = []
    print(f"Loading news from {splits}...")
    for split in splits:
        news_path = base_path / split / "news.tsv"
        if news_path.exists():
            print(f"  - Loading {news_path}")
            df = _load_news_df(str(news_path))
            dfs.append(df)
        else:
            print(f"  - {news_path} not found")

    if not dfs:
        print("No news files found!")
        return

    full_news_df = pd.concat(dfs, ignore_index=True)
    # Deduplicate by ID, keeping the last occurrence
    full_news = full_news_df.drop_duplicates(subset=["id"], keep="last")
    print(f"Total unique news articles: {len(full_news)} (from {len(full_news_df)} raw entries)")

    # Load Mappings (from train dir)
    train_dir = base_path / "train"
    category_mapping = _load_mapping(train_dir / "category2int.tsv")
    subcategory_mapping = _load_mapping(train_dir / "subcategory2int.tsv")

    # Preprocessing text
    full_news["title"] = full_news["title"].fillna("")
    full_news["abstract"] = full_news["abstract"].fillna("")
    if "body" not in full_news.columns:
        full_news["body"] = ""
    full_news["body"] = full_news["body"].fillna("")

    # Map categories and subcategories
    if category_mapping:
        full_news["category"] = full_news["category"].astype(str).str.strip().map(category_mapping)
    if subcategory_mapping:
        full_news["subcategory"] = full_news["subcategory"].astype(str).str.strip().map(subcategory_mapping)

    full_news["category"] = pd.to_numeric(full_news["category"], errors='coerce').fillna(0).astype(int)
    full_news["subcategory"] = pd.to_numeric(full_news["subcategory"], errors='coerce').fillna(0).astype(int)

    # Tokenizer
    model_name = dataset_cfg["pretrained_model_name"]
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Determine max length
    max_len = (
        dataset_cfg.get("num_words_title", 20) + 
        dataset_cfg.get("num_words_abstract", 50) + 
        dataset_cfg.get("num_words_body", 0)
    )
    print(f"Max sequence length: {max_len}")

    tokenized_data: Dict[str, Any] = {}

    # Prepare Padding Entry
    pad_token = tokenizer(
        "",
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    # Store padding news entry
    # Note: Pad token ID is usually 0 for BERT-like models, but we use the tokenizer's output to be sure.
    tokenized_data["PADDED_NEWS"] = {
        "input_ids": pad_token["input_ids"],        # [1, seq_len]
        "attention_mask": pad_token["attention_mask"], # [1, seq_len]
        "category": torch.tensor(0).long(),
        "subcategory": torch.tensor(0).long()
    }

    print("Tokenizing news articles...")
    news_records = full_news.to_dict('records')
    
    # Batch processing is generally faster but simple iteration is robust enough for this scale
    for row in tqdm(news_records):
        text = f"{row['title']} {row['abstract']} {row['body']}".strip()
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        tokenized_data[str(row['id'])] = {
            "input_ids": encoded["input_ids"],  # [1, seq_len]
            "attention_mask": encoded["attention_mask"],
            "category": torch.tensor(row['category']).long(),
            "subcategory": torch.tensor(row['subcategory']).long()
        }

    # Determine output path
    if output_path:
        save_path = Path(output_path)
    else:
        save_path = base_path / "news_tokenized.pt"

    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {len(tokenized_data)} entries to {save_path}...")
    torch.save(tokenized_data, save_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize news articles for fast loading during training.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        help="Dataset name as defined in config.yaml (e.g., 'mind')."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Optional output path for the tokenized file. Defaults to 'news_tokenized.pt' in the dataset directory."
    )
    
    args = parser.parse_args()
    
    try:
        tokenize_news(args.dataset, args.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
