import argparse
import yaml
import pandas as pd
from ast import literal_eval
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path
# No longer need argument parser - config is passed directly

class AdressaNewsDataset(Dataset):
    """
    Load news for evaluation for ebnerd dataset.
    """
    def __init__(self, news_path, pretrained_model_name, model_name, dataset_type=None, num_words_title=20, num_words_abstract=50):
        """
        Args:
            news_path: path to news file
            pretrained_model_name: name of the pretrained model
            model_name: name of the model
            dataset_type: type of dataset (e.g., 'mind', 'ebnerd')
            num_words_title: maximum number of words in title
            num_words_abstract: maximum number of words in abstract
        """
        super(AdressaNewsDataset, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        
        # Extract dataset type from news_path if not provided
        if dataset_type is None:
            # Extract from path like "data/ebnerd/val/news.tsv" -> "ebnerd"
            path_parts = news_path.split('/')
            if 'data' in path_parts:
                data_idx = path_parts.index('data')
                if data_idx + 1 < len(path_parts):
                    dataset_type = path_parts[data_idx + 1]
        
        self.dataset = dataset_type
        self.num_words_title = num_words_title
        self.num_words_abstract = num_words_abstract

        # Resolve dataset root so we can reuse mapping files emitted during preprocessing
        news_path_obj = Path(news_path)
        self.dataset_root = news_path_obj.parent.parent

        # Read news file with ebnerd/adressa format (no header, tab-separated)
        self.news = pd.read_table(
            news_path,
            header=None,
            names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'entities', 'entity_groups'],
            usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
        )
     
        self.news['title'] = self.news['title'].fillna('')
        self.news['abstract'] = self.news['abstract'].fillna('')
        self.news['category_text'] = self.news['category'].fillna('').astype(str)

        # Load category mapping from precomputed mapping file when available
        category2int_path = self.dataset_root / 'train' / 'category2int.tsv'
        if category2int_path.exists():
            category2int_df = pd.read_table(category2int_path, header=None, names=['category', 'int'])
            category2int = dict(category2int_df.values.tolist())
        else:
            category2int = {}

        self.news['category'] = self.news['category'].map(category2int)
        self.news['subcategory'] = self.news['subcategory'].map(category2int)
        
        self.news['category'] = self.news['category'].fillna(0)
        self.news['subcategory'] = self.news['subcategory'].fillna(0)

        # Store only raw fields for tokenization in __getitem__
        self.news_parsed = []
        for index, row in enumerate(self.news.itertuples()):
            text_parts = [row.category_text.strip(), row.title, row.abstract]
            text = ' '.join(part for part in text_parts if part)
            self.news_parsed.append({
                'id': row.id,
                'text': text,
                'category': row.category,
                'subcategory': row.subcategory
            })

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
    
class AdressaBehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation for ebnerd dataset, (user, time) pair as session
    """
    def __init__(self, behaviors_path, cold_threshold=0, num_clicked_news_a_user=30):
        """
        Args:
            behaviors_path: path to behaviors file
            cold_threshold: cold start threshold
            num_clicked_news_a_user: maximum number of clicked news per user
        """
        super(AdressaBehaviorsDataset, self).__init__()
        
        behaviors_path_obj = Path(behaviors_path)
        self.dataset_root = behaviors_path_obj.parent.parent

        # Read behaviors file with ebnerd format (no header, tab-separated)
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].astype(str).str.split()

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

        mapping_path = self.dataset_root / 'train' / 'user2int.tsv'
        if mapping_path.exists():
            mapping_df = pd.read_table(mapping_path, header=None, names=['user', 'int'])
            user2int = {str(row.user): int(row.int) for row in mapping_df.itertuples(index=False)}
            self.behaviors['user'] = self.behaviors['user'].astype(str).map(user2int).fillna(-1).astype(int)
        else:
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

class AdressaUserDataset(Dataset):
    """
    Load users for evaluation for ebnerd dataset, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path, dataset_type: str | None = None, num_clicked_news_a_user: int = 30):
        """
        Args:
            behaviors_path: path to behaviors file
        """
        super(AdressaUserDataset, self).__init__()
        
        # Infer dataset type from path if not provided
        if dataset_type is None:
            path_parts = behaviors_path.split('/')
            if 'data' in path_parts:
                data_idx = path_parts.index('data')
                if data_idx + 1 < len(path_parts):
                    dataset_type = path_parts[data_idx + 1]

        self.dataset = dataset_type
        self.num_clicked_news_a_user = num_clicked_news_a_user

        behaviors_path_obj = Path(behaviors_path)
        self.dataset_root = behaviors_path_obj.parent.parent

        # Read behaviors file with ebnerd/adressa format (no header, tab-separated)
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=[0, 1, 3],
                                       names=['id', 'user', 'clicked_news'])
        
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')

        # Map users to int ids using precomputed mapping if available, else build from file
        mapping_path = self.dataset_root / 'train' / 'user2int.tsv'
        if mapping_path.exists():
            mapping_df = pd.read_table(mapping_path, header=None, names=['user', 'int'])
            user2int = {str(row.user): int(row.int) for row in mapping_df.itertuples(index=False)}
            self.behaviors['user'] = self.behaviors['user'].astype(str).map(user2int).fillna(-1).astype(int)
        else:
            unique_users = self.behaviors['user'].unique()
            user2int = {user: idx for idx, user in enumerate(unique_users)}
            self.behaviors['user'] = self.behaviors['user'].map(user2int)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]

        item = {
            "id": row.user,
            "clicked_news": row.clicked_news.split()[-self.num_clicked_news_a_user:]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = self.num_clicked_news_a_user - len(item["clicked_news"])

        item["clicked_news_mask"] = [0] * repeated_times + [1] * len(item["clicked_news"])
        item["clicked_news"] = ['PADDED_NEWS'] * repeated_times + item["clicked_news"]

        return item
    
class AdressaBaseDataset(Dataset):
    """
    Base dataset for ebnerd training with negative sampling
    """
    def __init__(self, behaviors_path, news_path, config):
        super(AdressaBaseDataset, self).__init__()

        self.config = config
        self.dataset = config['_dataset']  # Get dataset from config
        dataset_root = Path(behaviors_path).parent.parent
        self.dataset_root = dataset_root
        self.tokenizer = AutoTokenizer.from_pretrained(config['datasets'][self.dataset]['pretrained_model_name'])

        # Read parsed behaviors file (with header, tab-separated)
        self.behaviors_parsed = pd.read_table(behaviors_path, sep='\t', names=['user', 'clicked_news', 'candidate_news', 'clicked'])

        self.behaviors_parsed['clicked_news'] = self.behaviors_parsed['clicked_news'].fillna(' ')

        mapping_path = dataset_root / 'train' / 'user2int.tsv'
        if mapping_path.exists():
            mapping_df = pd.read_table(mapping_path, header=None, names=['user', 'int'])
            user2int = {str(row.user): int(row.int) for row in mapping_df.itertuples(index=False)}
            self.behaviors_parsed['user'] = self.behaviors_parsed['user'].astype(str).map(user2int).fillna(0)
        else:
            user2int = dict(pd.read_table(f'data/{self.dataset}/train/user2int.tsv').values.tolist())
            self.behaviors_parsed['user'] = self.behaviors_parsed['user'].map(user2int)
            self.behaviors_parsed['user'] = self.behaviors_parsed['user'].fillna(0)

        self.behaviors_parsed = self.behaviors_parsed[self.behaviors_parsed['candidate_news'].notna()]

        self.news_parsed = pd.read_table(
            news_path,
            header=None,
            names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'entities', 'entity_groups'],
            usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
        )

        self.news_parsed['title'] = self.news_parsed['title'].fillna('')
        self.news_parsed['abstract'] = self.news_parsed['abstract'].fillna('')
        self.news_parsed['category_text'] = self.news_parsed['category'].fillna('').astype(str)

        category_mapping_path = dataset_root / 'train' / 'category2int.tsv'
        if category_mapping_path.exists():
            category_df = pd.read_table(category_mapping_path, header=None, names=['category', 'int'])
            category2int = dict(category_df.values.tolist())
        else:
            category2int = dict(pd.read_table(f'data/{self.dataset}/train/category2int.tsv').values.tolist())

        self.news_parsed['category'] = self.news_parsed['category'].map(category2int)
        self.news_parsed['subcategory'] = self.news_parsed['subcategory'].map(category2int)
        self.news_parsed['category'] = self.news_parsed['category'].fillna(0)
        self.news_parsed['subcategory'] = self.news_parsed['subcategory'].fillna(0)

        # Store only raw fields for tokenization in __getitem__
        self.news2raw = {}
        for index, row in enumerate(self.news_parsed.itertuples()):
            text_parts = [row.category_text.strip(), row.title, row.abstract]
            text = ' '.join(part for part in text_parts if part)
            self.news2raw[str(row.id)] = {
                'text': text,
                'category': row.category,
                'subcategory': row.subcategory
            }

        # Add explicit padding entry for consistency
        self.padding = {
            'text': '',
            'category': 0,
            'subcategory': 0
        }
        self.news2raw['PADDED_NEWS'] = self.padding
        self.padding_tokenized = None  # Will be set on first use

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]

        def get_tokenized(news_id):
            raw = self.news2raw.get(news_id, self.padding)
            tokenized = self.tokenizer(
                raw['text'],
                padding='max_length',
                truncation=True,
                max_length=self.config['datasets'][self.dataset]['num_words_title'] + self.config['datasets'][self.dataset]['num_words_abstract'],
                return_tensors='pt'
            )

            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'category': torch.tensor(raw['category']).long(),
                'subcategory': torch.tensor(raw['subcategory']).long(),
                'text': raw['text']
            }

        # Prepare padding tokenized if not already
        if self.padding_tokenized is None:
            self.padding_tokenized = get_tokenized('PADDED_NEWS')  # Use 'PADDED_NEWS' consistently

        # Get candidate news and clicked labels
        candidate_news_ids = row.candidate_news.split()
        clicked_labels = [int(x) for x in row.clicked.split()]
        
        # No shuffling, preserve original order
        item["candidate_news"] = [
            get_tokenized(x) for x in candidate_news_ids
        ]
        item["clicked"] = torch.tensor(clicked_labels).long()

        clicked_news_ids = row.clicked_news.split()[-self.config['datasets'][self.dataset]['num_clicked_news_a_user']:]
        item["clicked_news"] = [
            get_tokenized(x) for x in clicked_news_ids
        ]

        item["clicked_news_length"] = len(item["clicked_news"])
        repeated_times = self.config['datasets'][self.dataset]['num_clicked_news_a_user'] - len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news_mask"] = [0] * repeated_times + [1] * len(item["clicked_news"])
        item["clicked_news"] = [self.padding_tokenized] * repeated_times + item["clicked_news"]
        item["user"] = torch.tensor(row.user).long()
        
        return item 
