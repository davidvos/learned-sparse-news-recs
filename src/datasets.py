from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _infer_dataset_from_path(file_path: str) -> str | None:
    """Infer dataset name from a path that contains a data/<dataset>/ segment."""
    parts = Path(file_path).parts
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _load_mapping(mapping_path: Path) -> dict[str, int]:
    """Load mapping files like user2int.tsv or category2int.tsv when present."""
    if not mapping_path.exists():
        return {}
    mapping_df = pd.read_table(mapping_path, header=None, names=["key", "value"])
    return {str(row.key): int(row.value) for row in mapping_df.itertuples(index=False)}

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
    ) -> None:
        # Model name is accepted for backwards compatibility but unused now
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        inferred_dataset = dataset_type or _infer_dataset_from_path(news_path)
        if inferred_dataset is None:
            raise ValueError(f"Unable to infer dataset from path: {news_path}")
        self.dataset = inferred_dataset
        self.num_words_title = num_words_title
        self.num_words_abstract = num_words_abstract

        news_path_obj = Path(news_path)
        dataset_root = news_path_obj.parent.parent if news_path_obj.parent.name in {"train", "val", "test"} else news_path_obj.parent
        category_mapping = _load_mapping(dataset_root / "train" / "category2int.tsv")

        self.news = pd.read_table(
            news_path,
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
            usecols=["id", "category", "subcategory", "title", "abstract"],
        )

        self.news["title"] = self.news["title"].fillna("")
        self.news["abstract"] = self.news["abstract"].fillna("")

        if category_mapping:
            self.news["category"] = self.news["category"].map(category_mapping)
            self.news["subcategory"] = self.news["subcategory"].map(category_mapping)

        self.news["category"] = self.news["category"].fillna(0)
        self.news["subcategory"] = self.news["subcategory"].fillna(0)

        # Prepare cached representation for __getitem__ to avoid repeated DataFrame access
        self.news_parsed = []
        for row in self.news.itertuples():
            text = f"{row.title} {row.abstract}".strip()
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
            max_length=self.num_words_title + self.num_words_abstract,
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
    def __init__(self, behaviors_path, cold_threshold=0, num_clicked_news_a_user=30):
        """
        Args:
            behaviors_path: path to behaviors file
            cold_threshold: cold start threshold
            num_clicked_news_a_user: maximum number of clicked news per user
        """
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].str.split()

        self.cold_threshold = cold_threshold
        self.num_clicked_news_a_user = num_clicked_news_a_user
        
        if cold_threshold == 0:
            max_clicked = num_clicked_news_a_user
        else:
            max_clicked = cold_threshold

        # Truncate to max_clicked, no padding here
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].apply(
            lambda x: x[-max_clicked:]
        )
        self.behaviors['clicked_news_length'] = self.behaviors['clicked_news'].apply(lambda x: len(x))
        self.behaviors['clicked_news_mask'] = self.behaviors['clicked_news_length'].apply(lambda x: [1] * x)
        
        self.behaviors.impressions = self.behaviors.impressions.fillna(' ')
        self.behaviors.impressions = self.behaviors.impressions.str.split()

        unique_users = self.behaviors['user'].unique()
        user2int = {user: idx for idx, user in enumerate(unique_users)}
        self.behaviors['user'] = self.behaviors['user'].map(user2int)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user": row.user,
            "impressions": row.impressions,
            "clicked_news": row.clicked_news,
            "clicked_news_length": row.clicked_news_length,
            "clicked_news_mask": row.clicked_news_mask
        }
        return item

class BaseDataset(Dataset):
    """Dataset used during training for MIND-style datasets."""

    def __init__(self, behaviors_path: str, news_path: str, config: dict) -> None:
        super().__init__()

        self.config = config
        self.dataset = config["_dataset"]
        dataset_cfg = config["datasets"][self.dataset]
        self.tokenizer = AutoTokenizer.from_pretrained(dataset_cfg["pretrained_model_name"])

        behaviors_path_obj = Path(behaviors_path)
        dataset_root = behaviors_path_obj.parent.parent if behaviors_path_obj.parent.name in {"train", "val", "test"} else behaviors_path_obj.parent

        user_mapping = _load_mapping(dataset_root / "train" / "user2int.tsv")

        self.behaviors_parsed = pd.read_table(behaviors_path, sep="\t")
        self.behaviors_parsed["clicked_news"] = self.behaviors_parsed["clicked_news"].fillna(" ")
        if user_mapping:
            self.behaviors_parsed["user"] = (
                self.behaviors_parsed["user"].astype(str).map(user_mapping).fillna(0).astype(int)
            )
        else:
            unique_users = self.behaviors_parsed["user"].astype(str).unique()
            fallback_mapping = {user: idx for idx, user in enumerate(unique_users)}
            self.behaviors_parsed["user"] = self.behaviors_parsed["user"].astype(str).map(fallback_mapping)

        category_mapping = _load_mapping(dataset_root / "train" / "category2int.tsv")
        self.news_parsed = pd.read_table(
            news_path,
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
            usecols=["id", "category", "subcategory", "title", "abstract"],
        )

        self.news_parsed["title"] = self.news_parsed["title"].fillna("")
        self.news_parsed["abstract"] = self.news_parsed["abstract"].fillna("")

        if category_mapping:
            self.news_parsed["category"] = self.news_parsed["category"].map(category_mapping)
            self.news_parsed["subcategory"] = self.news_parsed["subcategory"].map(category_mapping)

        self.news_parsed["category"] = self.news_parsed["category"].fillna(0)
        self.news_parsed["subcategory"] = self.news_parsed["subcategory"].fillna(0)

        # Cache news features so __getitem__ can quickly assemble tensors
        self.news2raw: dict[str, dict[str, object]] = {}
        for row in self.news_parsed.itertuples():
            text = f"{row.title} {row.abstract}".strip()
            self.news2raw[str(row.id)] = {
                "text": text,
                "category": int(row.category),
                "subcategory": int(row.subcategory),
            }

        self.padding = {"text": "", "category": 0, "subcategory": 0}
        self.news2raw["PADDED_NEWS"] = self.padding
        self.padding_tokenized = None  # Lazy init to avoid extra tokenization work

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]

        def get_tokenized(news_id: str) -> dict[str, torch.Tensor]:
            raw = self.news2raw.get(news_id, self.padding)
            tokenized = self.tokenizer(
                raw["text"],
                padding="max_length",
                truncation=True,
                max_length=dataset_cfg["num_words_title"] + dataset_cfg["num_words_abstract"],
                return_tensors="pt",
            )
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "category": torch.tensor(raw["category"]).long(),
                "subcategory": torch.tensor(raw["subcategory"]).long(),
                "text": raw["text"],
            }

        # Prepare padding tokenized if not already
        if self.padding_tokenized is None:
            self.padding_tokenized = get_tokenized('PADDED_NEWS')  # Use 'PADDED_NEWS' consistently

        # Get candidate news and clicked labels
        candidate_news_ids = str(row.candidate_news).split()
        clicked_labels = [int(x) for x in str(row.clicked).split()]
        
        # No shuffling, preserve original order
        item["candidate_news"] = [
            get_tokenized(x) for x in candidate_news_ids
        ]
        item["clicked"] = torch.tensor(clicked_labels).long()

        history_limit = dataset_cfg["num_clicked_news_a_user"]
        clicked_news_ids = str(row.clicked_news).split()[-history_limit:]
        item["clicked_news"] = [
            get_tokenized(x) for x in clicked_news_ids
        ]

        item["clicked_news_length"] = len(item["clicked_news"])
        repeated_times = history_limit - len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news_mask"] = [0] * repeated_times + [1] * len(item["clicked_news"])
        item["clicked_news"] = [self.padding_tokenized] * repeated_times + item["clicked_news"]
        item["user"] = torch.tensor(row.user).long()

        return item
