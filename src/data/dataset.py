import random
from pathlib import Path
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer

def _infer_dataset_from_path(file_path: str) -> str | None:
    parts = Path(file_path).parts
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None

def _load_mapping(mapping_path: Path) -> dict[str, int]:
    if not mapping_path.exists():
        return {}
    mapping_df = pd.read_table(mapping_path, header=None, names=["key", "value"])
    # Strip whitespace from keys to be robust
    return {str(row.key).strip(): int(row.value) for row in mapping_df.itertuples(index=False)}
    
def _load_news_df(news_path: str) -> pd.DataFrame:
    with open(news_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        num_cols = len(first_line.split('\t'))
    
    if num_cols == 9:
        # EBNeRD style with 'body'
        names = ["id", "category", "subcategory", "title", "abstract", "body", "url", "title_entities", "abstract_entities"]
        usecols = ["id", "category", "subcategory", "title", "abstract", "body"]
    else:
        # MIND style (8 columns or fallback)
        names = ["id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
        usecols = ["id", "category", "subcategory", "title", "abstract"]
        
    df = pd.read_table(
        news_path,
        names=names,
        usecols=usecols,
    )
    if "body" not in df.columns:
        df["body"] = ""
    return df

class NewsDataset(Dataset):
    """Load news articles for evaluation splits."""

    def __init__(
        self,
        news_path: str,
        pretrained_model_name: str,
        model_name: str | None = None,
        dataset_type: str | None = None,
        num_words_title: int = 20,
        num_words_abstract: int = 50,
        num_words_body: int = 0,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        inferred_dataset = dataset_type or _infer_dataset_from_path(news_path)
        if inferred_dataset is None:
            raise ValueError(f"Unable to infer dataset from path: {news_path}")
        self.dataset = inferred_dataset
        self.num_words_title = num_words_title
        self.num_words_abstract = num_words_abstract
        self.num_words_body = num_words_body
        self.model_name = model_name

        news_path_obj = Path(news_path)
        dataset_root = news_path_obj.parent.parent if news_path_obj.parent.name in {"train", "val", "test"} else news_path_obj.parent
        category_mapping = _load_mapping(dataset_root / "train" / "category2int.tsv")
        subcategory_mapping = _load_mapping(dataset_root / "train" / "subcategory2int.tsv")

        self.news = _load_news_df(news_path)

        self.news["title"] = self.news["title"].fillna("")
        self.news["abstract"] = self.news["abstract"].fillna("")
        self.news["body"] = self.news["body"].fillna("")

        if category_mapping:
            self.news["category"] = self.news["category"].astype(str).str.strip().map(category_mapping)
        if subcategory_mapping:
            self.news["subcategory"] = self.news["subcategory"].astype(str).str.strip().map(subcategory_mapping)

        self.news["category"] = pd.to_numeric(self.news["category"], errors='coerce').fillna(0).astype(int)
        self.news["subcategory"] = pd.to_numeric(self.news["subcategory"], errors='coerce').fillna(0).astype(int)

        self.news_parsed = []
        for row in self.news.itertuples():
            if self.model_name == 'star':
                # Align with STARDataset._build_candidate_text
                title = str(row.title)
                abstract = str(row.abstract)
                
                # Word-based truncation
                title_words = title.split()[:self.num_words_title]
                title = " ".join(title_words)
                
                if abstract:
                    abstract_words = abstract.split()[:self.num_words_abstract]
                    abstract = " ".join(abstract_words)
                
                parts = ["[NEWS]", title]
                if abstract:
                    parts.append(abstract)
                text = ". ".join(parts)
            else:
                text = f"{row.title} {row.abstract} {row.body}".strip()
                
            self.news_parsed.append(
                {
                    "id": row.id,
                    "text": text,
                    "category": int(row.category),
                    "subcategory": int(row.subcategory),
                }
            )

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news_parsed[idx]
        tokenized_text = self.tokenizer(
            item['text'],
            padding='max_length',
            truncation=True,
            max_length=self.num_words_title + self.num_words_abstract + self.num_words_body,
            return_tensors='pt'
        )
        return {
            'id': item['id'],
            'input_ids': tokenized_text['input_ids'],
            'attention_mask': tokenized_text['attention_mask'],
            'category': torch.tensor(item['category']).long(),
            'subcategory': torch.tensor(item['subcategory']).long(),
            'text': item['text']
        }
    
class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path, cold_threshold=0, num_clicked_news_a_user=30, min_history=0, max_history=0, actual_history=False):
        super(BehaviorsDataset, self).__init__()
        
        is_parsed = "behaviors_parsed" in behaviors_path
        header = 0 if is_parsed else None
        
        if is_parsed:
            names=["impression_id", "user", "time", "clicked_news", "impressions", "candidate_news", "clicked"]
            use_cols = range(7)
        else:
            names=['impression_id', 'user', 'time', 'clicked_news', 'impressions']
            use_cols = range(5)

        self.behaviors = pd.read_table(behaviors_path,
                                       header=header,
                                       usecols=use_cols,
                                       names=names)
        
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].str.split()

        self.cold_threshold = cold_threshold
        self.num_clicked_news_a_user = num_clicked_news_a_user
        self.min_history = min_history
        self.max_history = max_history
        self.actual_history = actual_history

        if self.min_history > 0 or self.max_history > 0:
            if self.min_history > 0:
                limit = self.cold_threshold if self.cold_threshold > 0 and not self.actual_history else self.num_clicked_news_a_user
                self.behaviors = self.behaviors[self.behaviors['clicked_news'].apply(lambda x: min(len(x), limit) >= self.min_history)]
            
            if self.max_history > 0:
                self.behaviors = self.behaviors[self.behaviors['clicked_news'].apply(lambda x: len(x) <= self.max_history)]

        self.behaviors.impressions = self.behaviors.impressions.fillna(' ')
        self.behaviors.impressions = self.behaviors.impressions.str.split()

        behaviors_path_obj = Path(behaviors_path)
        dataset_root = (
            behaviors_path_obj.parent.parent 
            if behaviors_path_obj.parent.name in {"train", "val", "test"} 
            else behaviors_path_obj.parent
        )
        user_mapping = _load_mapping(dataset_root / "train" / "user2int.tsv")

        if user_mapping:
            self.behaviors["user"] = (
                self.behaviors["user"].astype(str).map(user_mapping).fillna(0).astype(int)
            )
        else:
            unique_users = self.behaviors['user'].unique()
            fallback_mapping = {user: idx for idx, user in enumerate(unique_users)}
            self.behaviors['user'] = self.behaviors['user'].map(fallback_mapping)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        clicked_news = row.clicked_news
        actual_len = len(clicked_news)
        
        if self.actual_history:
            truncated_history = clicked_news[-self.num_clicked_news_a_user:]
        elif self.cold_threshold > 0:
            limit = min(actual_len, self.cold_threshold)
            truncated_history = clicked_news[-limit:] if limit > 0 else []
        else:
            truncated_history = clicked_news[-self.num_clicked_news_a_user:]

        clicked_news_length = len(truncated_history)
        clicked_news_mask = [1] * clicked_news_length

        item = {
            "impression_id": row.impression_id if "impression_id" in self.behaviors.columns else idx,
            "user": row.user,
            "impressions": row.impressions,
            "clicked_news": truncated_history,
            "clicked_news_length": clicked_news_length,
            "clicked_news_mask": clicked_news_mask
        }
        return item

class BaseDataset(Dataset):
    """Dataset used during training for MIND-style datasets. Optimized for H100."""

    def __init__(self, behaviors_path: str, news_path: str, config: dict, negative_sampling_ratio: int = 0) -> None:
        super().__init__()

        self.config = config
        self.dataset = config["_dataset"]
        dataset_cfg = config["datasets"][self.dataset]
        self.dataset_cfg = dataset_cfg
        self.tokenizer = AutoTokenizer.from_pretrained(dataset_cfg["pretrained_model_name"])
        self.negative_sampling_ratio = negative_sampling_ratio

        behaviors_path_obj = Path(behaviors_path)
        dataset_root = behaviors_path_obj.parent.parent if behaviors_path_obj.parent.name in {"train", "val", "test"} else behaviors_path_obj.parent

        user_mapping = _load_mapping(dataset_root / "train" / "user2int.tsv")

        self.behaviors_parsed = pd.read_table(
            behaviors_path,
            sep="\t",
            header=0,
            names=["user", "clicked_news", "candidate_news", "clicked"],
            dtype=str,
        )
        self.behaviors_parsed["clicked_news"] = self.behaviors_parsed["clicked_news"].fillna(" ")
        
        initial_count = len(self.behaviors_parsed)
        self.behaviors_parsed = self.behaviors_parsed[
            self.behaviors_parsed["clicked_news"].str.strip().str.len() > 0
        ]
        filtered_count = initial_count - len(self.behaviors_parsed)
        if filtered_count > 0:
            print(f"Filtered {filtered_count} samples with empty click history")
        
        if user_mapping:
            self.behaviors_parsed["user"] = (
                self.behaviors_parsed["user"].astype(str).map(user_mapping).fillna(0).astype(int)
            )
        else:
            unique_users = self.behaviors_parsed["user"].astype(str).unique()
            fallback_mapping = {user: idx for idx, user in enumerate(unique_users)}
            self.behaviors_parsed["user"] = self.behaviors_parsed["user"].astype(str).map(fallback_mapping)


        category_mapping = _load_mapping(dataset_root / "train" / "category2int.tsv")
        subcategory_mapping = _load_mapping(dataset_root / "train" / "subcategory2int.tsv")
        self.news_parsed = _load_news_df(news_path)

        self.news_parsed["title"] = self.news_parsed["title"].fillna("")
        self.news_parsed["abstract"] = self.news_parsed["abstract"].fillna("")
        self.news_parsed["body"] = self.news_parsed["body"].fillna("")

        if category_mapping:
            self.news_parsed["category"] = self.news_parsed["category"].astype(str).str.strip().map(category_mapping)
        if subcategory_mapping:
            self.news_parsed["subcategory"] = self.news_parsed["subcategory"].astype(str).str.strip().map(subcategory_mapping)

        self.news_parsed["category"] = pd.to_numeric(self.news_parsed["category"], errors='coerce').fillna(0).astype(int)
        self.news_parsed["subcategory"] = pd.to_numeric(self.news_parsed["subcategory"], errors='coerce').fillna(0).astype(int)

        # Cache raw metadata for fast lookup (fallback if pretokenized missing)
        self.news2raw: dict[str, dict[str, object]] = {}
        for row in self.news_parsed.itertuples():
            self.news2raw[str(row.id)] = {
                "text": f"{row.title} {row.abstract} {row.body}".strip() if pd.notna(row.title) else "",
                "category": int(row.category) if pd.notna(row.category) else 0,
                "subcategory": int(row.subcategory) if pd.notna(row.subcategory) else 0,
            }

        self.padding = {"text": "", "category": 0, "subcategory": 0}
        self.news2raw["PADDED_NEWS"] = self.padding
        self.padding_tokenized = None
        
        self.pad_histories = dataset_cfg.get("pad_clicked_history", False)
        self.pad_candidates = dataset_cfg.get("pad_candidate_news", False)
        candidate_limit_cfg = dataset_cfg.get("max_candidate_news")
        self.candidate_limit = candidate_limit_cfg if candidate_limit_cfg and candidate_limit_cfg > 0 else None

        self.pretokenized_news = None
        self.pretokenized_news_path = None
        
        news_path_obj = Path(news_path)
        potential_paths = [
            news_path_obj.parent / "news_tokenized.pt", 
            dataset_root / "news_tokenized.pt"
        ]
        
        for p in potential_paths:
            if p.exists():
                print(f"Found pre-tokenized news at {p}. Will lazy-load in workers.")
                self.pretokenized_news_path = p
                break
        
        if self.pretokenized_news_path is None:
             print("Warning: No pre-tokenized news found. Falling back to slow on-the-fly tokenization.")

        # History dropout augmentation
        self.enable_history_dropout = config['training'].get('history_dropout', False)
        self.dropout_p = config['training'].get('history_dropout_prob', 0.4)

    def _ensure_pretokenized_loaded(self):
        """Lazy loader called by each worker process to load its own copy."""
        if self.pretokenized_news is None and self.pretokenized_news_path is not None:
            # Load into RAM only when the worker starts processing
            self.pretokenized_news = torch.load(self.pretokenized_news_path, map_location='cpu')

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        self._ensure_pretokenized_loaded()
        
        row = self.behaviors_parsed.iloc[idx]

        # Helper to get tokens (Dict of Tensors)
        def get_tokens(news_id: str):
            if self.pretokenized_news is not None:
                # Fast Path: Hash map lookup
                if news_id in self.pretokenized_news:
                    return self.pretokenized_news[news_id]
                else:
                    return self.pretokenized_news["PADDED_NEWS"]
            
            # Slow Path: Tokenize on CPU
            raw = self.news2raw.get(news_id, self.padding)
            tokenized = self.tokenizer(
                raw["text"],
                padding="max_length",
                truncation=True,
                max_length=self.dataset_cfg.get("num_words_title", 20) + self.dataset_cfg.get("num_words_abstract", 50),
                return_tensors="pt",
            )
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "category": torch.tensor(raw["category"]).long(),
                "subcategory": torch.tensor(raw["subcategory"]).long(),
            }

        # Initialize padding if needed
        if self.padding_tokenized is None:
            self.padding_tokenized = get_tokens('PADDED_NEWS')

        # --- 2. Candidate Logic (Optimized) ---
        candidate_news_ids = str(row.candidate_news).split()
        clicked_labels = [int(x) for x in str(row.clicked).split()]

        # Negative Sampling
        if self.negative_sampling_ratio > 0:
            positives = [(nid, label) for nid, label in zip(candidate_news_ids, clicked_labels) if label == 1]
            negatives = [(nid, label) for nid, label in zip(candidate_news_ids, clicked_labels) if label == 0]
            
            sampled_ids = []
            sampled_labels = []
            for pos_nid, pos_label in positives:
                sampled_ids.append(pos_nid)
                sampled_labels.append(pos_label)
                if negatives:
                    selected_negs = random.sample(negatives, min(len(negatives), self.negative_sampling_ratio))
                    for neg_nid, neg_label in selected_negs:
                        sampled_ids.append(neg_nid)
                        sampled_labels.append(neg_label)
            
            combined = list(zip(sampled_ids, sampled_labels))
            random.shuffle(combined)
            candidate_news_ids, clicked_labels = zip(*combined) if combined else ([], [])

        if self.candidate_limit is not None:
            candidate_news_ids = candidate_news_ids[: self.candidate_limit]
            clicked_labels = clicked_labels[: self.candidate_limit]

        if not candidate_news_ids:
            candidate_news_ids = ["PADDED_NEWS"]
            clicked_labels = [0]
        
        # *** Optimization: Stack Tensors Immediately ***
        # Instead of returning [Dict, Dict...], return Dict[Tensor(Batch)]
        # UPDATE: The model expects List[Dict], so we must format it that way or change the model.
        # Changing the model is risky. Let's return List[Dict] but backed by the same data?
        # Actually, if we want speed, we should change the model to accept the stacked tensor.
        # BUT, the quick fix for the user is to return the List[Dict] structure they expect.
        
        # ... Wait, the user's manual change ALREADY implemented the stacking in __getitem__.
        # And their `collate_fn` (which they also modified?) or the default one handles it?
        # The user's trace shows `LSNR.forward` failing on `candidate_news`.
        # The trace says: `torch.cat([news['input_ids'] for news in candidate_news], dim=0)`
        # This implies `candidate_news` is iterable, but the items inside are strings?
        # "TypeError: string indices must be integers, not 'str'" => `news` is a string.
        # This means `candidate_news` is a list of strings (likely keys like "input_ids")? 
        # OR `candidate_news` is a Dictionary, and iterating over it yields keys (strings).
        
        # Yes! In the user's modified code, `item["candidate_news"]` is a Dict[str, Tensor].
        # In `LSNR.forward`, it iterates `for news in candidate_news`.
        # If `candidate_news` is a Dict, `news` becomes the key (e.g. "input_ids").
        # Then `news['input_ids']` tries to index the string key "input_ids" with "input_ids" -> TypeError.
        
        # FIX: We need to unpack the optimized Dict[Tensor] back into List[Dict] 
        # OR update the model to handle Dict[Tensor].
        # Given we want to avoid changing model code if possible, let's unpack in __getitem__ 
        # or collate.
        
        # Actually, let's just make __getitem__ return the List[Dict] structure. 
        # It's slightly less efficient than passing pure tensors but safe.
        
        item_candidates = []
        for nid in candidate_news_ids:
            t = get_tokens(nid)
            # Ensure input_ids/mask are [1, seq_len] for model compatibility
            ids = t['input_ids']
            if ids.dim() == 1: ids = ids.unsqueeze(0)
            
            mask = t['attention_mask']
            if mask.dim() == 1: mask = mask.unsqueeze(0)

            item_candidates.append({
                "input_ids": ids,
                "attention_mask": mask,
                "category": t['category'],
                "subcategory": t['subcategory'],
            })

        item_clicked_labels = torch.tensor(clicked_labels).long()

        # --- 3. History Logic (Optimized) ---
        # ... (Same logic for history)
        
        history_limit = self.dataset_cfg["num_clicked_news_a_user"]
        clicked_news_ids = str(row.clicked_news).split()[-history_limit:]

        # History Dropout
        if self.enable_history_dropout and len(clicked_news_ids) > 1:
            if random.random() < self.dropout_p:
                new_len = random.randint(1, len(clicked_news_ids))
                clicked_news_ids = clicked_news_ids[-new_len:]

        item_history = []
        for nid in clicked_news_ids:
            t = get_tokens(nid)
            # Ensure proper shape [1, seq_len]
            ids = t['input_ids']
            if ids.dim() == 1: ids = ids.unsqueeze(0)
            
            mask = t['attention_mask']
            if mask.dim() == 1: mask = mask.unsqueeze(0)

            item_history.append({
                "input_ids": ids,
                "attention_mask": mask,
                "category": t['category'],
                "subcategory": t['subcategory'],
            })
            
        actual_len = len(item_history)
        
        # Pad history manually if needed
        if self.pad_histories and actual_len < history_limit:
            pad_needed = history_limit - actual_len
            
            # Prepare padding token with correct shape
            pad_token = {}
            for k, v in self.padding_tokenized.items():
                if isinstance(v, torch.Tensor):
                    # Ensure padding tokens are also [1, seq_len] if needed
                    if k in ["input_ids", "attention_mask"] and v.dim() == 1:
                        pad_token[k] = v.unsqueeze(0).clone()
                    else:
                        pad_token[k] = v.clone()
                else:
                    pad_token[k] = v

            for _ in range(pad_needed):
                item_history.append({
                    k: v.clone() if isinstance(v, torch.Tensor) else v 
                    for k,v in pad_token.items()
                })
            
            mask_vec = torch.tensor([1] * actual_len + [0] * pad_needed).long()
        else:
            mask_vec = torch.ones(actual_len, dtype=torch.long)


        return {
            "candidate_news": item_candidates, # List[Dict]
            "clicked": item_clicked_labels,    # Tensor
            "clicked_news": item_history,      # List[Dict]
            "clicked_news_length": actual_len,
            "clicked_news_mask": mask_vec,
            "user": torch.tensor(row.user).long()
        }

    def collate_fn(self, batch: list[dict[str, object]]) -> dict[str, object]:
        # Batch: List of samples (dicts)
        # Each sample has "candidate_news": List[Dict[str, Tensor]] which varies in length
        
        # 1. Pad candidate_news to max length in batch
        max_cands = 0
        for sample in batch:
            max_cands = max(max_cands, len(sample["candidate_news"]))
            
        for sample in batch:
            cands = sample["candidate_news"]
            curr_len = len(cands)
            diff = max_cands - curr_len
            
            if diff > 0:
                # Append padding tokens
                # We reuse self.padding_tokenized, but must clone tensors and ensure dim matches
                # Note: self.padding_tokenized has [seq_len] tensors if from .pt, or [1, seq_len] if we fixed it?
                # In __getitem__ we ensured items in list are [1, seq_len].
                # So we should ensure padding is also [1, seq_len].
                
                pad_token = {}
                for k, v in self.padding_tokenized.items():
                    if isinstance(v, torch.Tensor):
                        # Force [1, seq_len] if needed to match __getitem__ output
                        if k in ["input_ids", "attention_mask"] and v.dim() == 1:
                            pad_token[k] = v.unsqueeze(0).clone()
                        else:
                            pad_token[k] = v.clone()
                    else:
                        pad_token[k] = v

                for _ in range(diff):
                    cands.append({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in pad_token.items()})
                
                # Pad labels
                sample["clicked"] = torch.cat([sample["clicked"], torch.zeros(diff, dtype=sample["clicked"].dtype)])

        return default_collate(batch)


class STARDataset(Dataset):
    """
    Dataset for STAR model that returns concatenated text strings.
    
    Key differences from BaseDataset:
    - User representation: Concatenated titles of clicked news (text string)
    - Candidate representation: "[Category] Title. Abstract" (text string)
    - Returns tokenized text inputs instead of per-article token dicts
    """
    
    def __init__(self, behaviors_path: str, news_path: str, config: dict, negative_sampling_ratio: int = 0) -> None:
        super().__init__()
        
        self.config = config
        self.dataset = config["_dataset"]
        dataset_cfg = config["datasets"][self.dataset]
        self.dataset_cfg = dataset_cfg
        self.model_cfg = config.get("_model_config", {})
        
        # STAR-specific config
        self.max_history_titles = self.model_cfg.get('max_history_titles', 10)
        self.max_user_tokens = self.model_cfg.get('max_user_tokens', 256)
        self.max_candidate_tokens = (
            dataset_cfg.get('num_words_title', 30) + 
            dataset_cfg.get('num_words_abstract', 50)
        )
        
        # Use SPLADE tokenizer by default, fall back to dataset's tokenizer
        pretrained_name = self.model_cfg.get(
            'pretrained_model_name',
            dataset_cfg.get('pretrained_model_name', 'naver/splade-cocondenser-ensembledistil')
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.negative_sampling_ratio = negative_sampling_ratio
        
        behaviors_path_obj = Path(behaviors_path)
        dataset_root = behaviors_path_obj.parent.parent if behaviors_path_obj.parent.name in {"train", "val", "test"} else behaviors_path_obj.parent
        
        user_mapping = _load_mapping(dataset_root / "train" / "user2int.tsv")
        category_mapping = _load_mapping(dataset_root / "train" / "category2int.tsv")
        subcategory_mapping = _load_mapping(dataset_root / "train" / "subcategory2int.tsv")
        
        # Reverse mappings for text construction
        self.int2category = {v: k for k, v in category_mapping.items()} if category_mapping else {}
        self.int2subcategory = {v: k for k, v in subcategory_mapping.items()} if subcategory_mapping else {}
        
        # Load behaviors
        self.behaviors_parsed = pd.read_table(
            behaviors_path,
            sep="\t",
            header=0,
            names=["user", "clicked_news", "candidate_news", "clicked"],
            dtype=str,
        )
        self.behaviors_parsed["clicked_news"] = self.behaviors_parsed["clicked_news"].fillna(" ")
        
        # Filter empty histories
        initial_count = len(self.behaviors_parsed)
        self.behaviors_parsed = self.behaviors_parsed[
            self.behaviors_parsed["clicked_news"].str.strip().str.len() > 0
        ]
        filtered_count = initial_count - len(self.behaviors_parsed)
        if filtered_count > 0:
            print(f"[STAR] Filtered {filtered_count} samples with empty click history")
        
        if user_mapping:
            self.behaviors_parsed["user"] = (
                self.behaviors_parsed["user"].astype(str).map(user_mapping).fillna(0).astype(int)
            )
        
        # Load news and build text lookup
        self.news_df = _load_news_df(news_path)
        self.news_df["title"] = self.news_df["title"].fillna("")
        self.news_df["abstract"] = self.news_df["abstract"].fillna("")
        self.news_df["body"] = self.news_df["body"].fillna("")
        
        if category_mapping:
            self.news_df["category"] = self.news_df["category"].astype(str).str.strip().map(category_mapping)
        if subcategory_mapping:
            self.news_df["subcategory"] = self.news_df["subcategory"].astype(str).str.strip().map(subcategory_mapping)
        
        self.news_df["category"] = pd.to_numeric(self.news_df["category"], errors='coerce').fillna(0).astype(int)
        self.news_df["subcategory"] = pd.to_numeric(self.news_df["subcategory"], errors='coerce').fillna(0).astype(int)
        
        # Build news lookup: id -> {title, abstract, category_name, subcategory_name}
        self.news2data: dict[str, dict] = {}
        for row in self.news_df.itertuples():
            cat_name = self.int2category.get(row.category, "")
            subcat_name = self.int2subcategory.get(row.subcategory, "")
            self.news2data[str(row.id)] = {
                "title": row.title,
                "abstract": row.abstract,
                "body": getattr(row, "body", ""),  # Safely get body
                "category": cat_name,
                "subcategory": subcat_name,
            }
        
        # Padding entry
        self.news2data["PADDED_NEWS"] = {
            "title": "",
            "abstract": "",
            "body": "",
            "category": "",
            "subcategory": "",
        }
        
        # History dropout
        self.enable_history_dropout = config['training'].get('history_dropout', False)
        self.dropout_p = config['training'].get('history_dropout_prob', 0.4)

        # Load News Scores for IDF selection
        self.news_scores = {}
        scores_path = dataset_root / "news_scores.tsv"
        if scores_path.exists():
            print(f"[STAR] Loading news scores from {scores_path}")
            scores_df = pd.read_table(scores_path, names=["id", "score"])
            self.news_scores = dict(zip(scores_df["id"].astype(str), scores_df["score"]))
        else:
            print(f"[STAR] Warning: No news scores found at {scores_path}. Using random/recent selection.")

    def _build_user_text(self, clicked_news_ids: list[str]) -> str:
        """
        Build user text from clicked news.
        
        Selection Strategy:
        1. If scores available: Select top-k highest scoring articles.
        2. Fallback: Select most recent interactions.
        
        Content: Title + Abstract
        """
        if not clicked_news_ids:
            return "[HISTORY]"

        if not clicked_news_ids:
            return "[HISTORY]"

        # Reverted to Recency-based selection (User Request)
        # Always select most recent first, regardless of score.
        sorted_ids = list(reversed(clicked_news_ids))

            
        # Greedily add articles (Title + Abstract) until limit
        prefix = "[HISTORY] "
        prefix_tokens = len(self.tokenizer.encode(prefix, add_special_tokens=False))
        available_tokens = self.max_user_tokens - prefix_tokens - 2 # [CLS] [SEP]
        
        # Word limits for truncation (defaults from config or standard)
        num_words_title = self.dataset_cfg.get("num_words_title", 20)
        num_words_abstract = self.dataset_cfg.get("num_words_abstract", 50)
        
        selected_texts = []
        current_tokens = 0
        
        for nid in sorted_ids:
            data = self.news2data.get(nid, self.news2data["PADDED_NEWS"])
            title = data["title"]
            abstract = data["abstract"]
            
            if not title:
                continue

            # Truncate to match LSNR/NRMS behavior (word-based)
            title_words = title.split()[:num_words_title]
            title = " ".join(title_words)
            
            if abstract:
                abstract_words = abstract.split()[:num_words_abstract]
                abstract = " ".join(abstract_words)
                
            # Content: "Title. Abstract."
            parts = [title]
            if abstract:
                parts.append(abstract)
                
            article_text = ". ".join(parts)
            if selected_texts:
                article_text += ". " # Separator
            
            # Check length with tokenizer
            article_tokens = len(self.tokenizer.encode(article_text, add_special_tokens=False))
            
            if current_tokens + article_tokens <= available_tokens:
                selected_texts.append(article_text)
                current_tokens += article_tokens
            else:
                # If full article doesn't fit, stop (greedy)
                # Could try to fit just title, but let's keep consistency for now
                continue
                
        if not selected_texts:
            return "[HISTORY]"
            
        return prefix + "".join(selected_texts)
    
    def _build_candidate_text(self, news_id: str) -> str:
        """
        Build candidate news text using title and abstract.
        Format: "[NEWS] Title. Abstract"
        """
        data = self.news2data.get(news_id, self.news2data["PADDED_NEWS"])
        
        num_words_title = self.dataset_cfg.get("num_words_title", 20)
        num_words_abstract = self.dataset_cfg.get("num_words_abstract", 50)
        
        title = data["title"]
        abstract = data["abstract"]
        
        # Truncate to match LSNR/NRMS behavior (word-based)
        title_words = title.split()[:num_words_title]
        title = " ".join(title_words)
        
        if abstract:
            abstract_words = abstract.split()[:num_words_abstract]
            abstract = " ".join(abstract_words)
            
        parts = ["[NEWS]", title]
        if abstract:
            parts.append(abstract)
            
        return ". ".join(parts)
    
    def __len__(self):
        return len(self.behaviors_parsed)
    
    def __getitem__(self, idx):
        row = self.behaviors_parsed.iloc[idx]
        
        # --- 1. Build User Text ---
        history_limit = self.dataset_cfg["num_clicked_news_a_user"]
        clicked_news_ids = str(row.clicked_news).split()[-history_limit:]
        
        # History dropout
        if self.enable_history_dropout and len(clicked_news_ids) > 1:
            if random.random() < self.dropout_p:
                new_len = random.randint(1, len(clicked_news_ids))
                clicked_news_ids = clicked_news_ids[-new_len:]
        
        user_text = self._build_user_text(clicked_news_ids)
        user_tokenized = self.tokenizer(
            user_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_user_tokens,
            return_tensors="pt",
        )
        
        # --- 2. Candidate Logic ---
        candidate_news_ids = str(row.candidate_news).split()
        clicked_labels = [int(x) for x in str(row.clicked).split()]
        
        # Negative Sampling
        if self.negative_sampling_ratio > 0:
            positives = [(nid, label) for nid, label in zip(candidate_news_ids, clicked_labels) if label == 1]
            negatives = [(nid, label) for nid, label in zip(candidate_news_ids, clicked_labels) if label == 0]
            
            sampled_ids = []
            sampled_labels = []
            for pos_nid, pos_label in positives:
                sampled_ids.append(pos_nid)
                sampled_labels.append(pos_label)
                if negatives:
                    selected_negs = random.sample(negatives, min(len(negatives), self.negative_sampling_ratio))
                    for neg_nid, neg_label in selected_negs:
                        sampled_ids.append(neg_nid)
                        sampled_labels.append(neg_label)
            
            combined = list(zip(sampled_ids, sampled_labels))
            random.shuffle(combined)
            candidate_news_ids, clicked_labels = zip(*combined) if combined else ([], [])
        
        if not candidate_news_ids:
            candidate_news_ids = ["PADDED_NEWS"]
            clicked_labels = [0]
        
        # Build candidate texts and tokenize
        item_candidates = []
        for nid in candidate_news_ids:
            cand_text = self._build_candidate_text(nid)
            cand_tokenized = self.tokenizer(
                cand_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_candidate_tokens,
                return_tensors="pt",
            )
            item_candidates.append({
                "input_ids": cand_tokenized["input_ids"].squeeze(0),  # [seq_len]
                "attention_mask": cand_tokenized["attention_mask"].squeeze(0),
            })
        
        return {
            "user_input": {
                "input_ids": user_tokenized["input_ids"].squeeze(0),  # [seq_len]
                "attention_mask": user_tokenized["attention_mask"].squeeze(0),
            },
            "candidate_news": item_candidates,  # List[Dict] for compatibility
            "clicked": torch.tensor(clicked_labels).long(),
            "clicked_news_length": len(clicked_news_ids),
            "user": torch.tensor(row.user).long() if hasattr(row, 'user') else torch.tensor(0).long(),
        }
    
    def collate_fn(self, batch: list[dict]) -> dict:
        """Custom collate for STAR - restructures candidates for model forward."""
        
        # Find max candidates in batch
        max_cands = max(len(sample["candidate_news"]) for sample in batch)
        
        # Pad candidates and labels to max_cands
        for sample in batch:
            cands = sample["candidate_news"]
            diff = max_cands - len(cands)
            
            if diff > 0:
                # Create padding token
                pad_tokenized = self.tokenizer(
                    "",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_candidate_tokens,
                    return_tensors="pt",
                )
                for _ in range(diff):
                    cands.append({
                        "input_ids": pad_tokenized["input_ids"].squeeze(0),
                        "attention_mask": pad_tokenized["attention_mask"].squeeze(0),
                    })
                
                # Pad labels
                sample["clicked"] = torch.cat([
                    sample["clicked"],
                    torch.zeros(diff, dtype=sample["clicked"].dtype)
                ])
        
        # Stack user inputs: [batch, seq_len]
        user_input = {
            "input_ids": torch.stack([s["user_input"]["input_ids"] for s in batch]),
            "attention_mask": torch.stack([s["user_input"]["attention_mask"] for s in batch]),
        }
        
        # Restructure candidates: list of num_candidates dicts, each with [batch, seq_len] tensors
        # Model expects candidate_input[i]['input_ids'] to be [batch, seq_len]
        candidate_news = []
        for cand_idx in range(max_cands):
            candidate_news.append({
                "input_ids": torch.stack([s["candidate_news"][cand_idx]["input_ids"] for s in batch]),
                "attention_mask": torch.stack([s["candidate_news"][cand_idx]["attention_mask"] for s in batch]),
            })
        
        # Stack other tensors
        clicked = torch.stack([s["clicked"] for s in batch])
        clicked_news_length = torch.tensor([s["clicked_news_length"] for s in batch])
        user = torch.stack([s["user"] for s in batch])
        
        return {
            "user_input": user_input,
            "candidate_news": candidate_news,
            "clicked": clicked,
            "clicked_news_length": clicked_news_length,
            "user": user,
        }