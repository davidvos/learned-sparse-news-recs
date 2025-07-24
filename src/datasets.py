import argparse
import yaml
import pandas as pd
from ast import literal_eval
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import get_arg_parser

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

args, unknown = get_arg_parser().parse_known_args()

class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """
    def __init__(self, news_path, pretrained_model_name, model_name):
        """
        Args:
            news_path: path to news file
            pretrained_model_name: name of the pretrained model
            model_name: name of the model
        """
        super(NewsDataset, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        self.news = pd.read_table(
            news_path,
            names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
            usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
        )
     
        self.news['title'] = self.news['title'].fillna('')
        self.news['abstract'] = self.news['abstract'].fillna('')

        category2int = dict(pd.read_table(f'data/{args.dataset}/train/category2int.tsv').values.tolist())
        self.news['category'] = self.news['category'].map(category2int)
        self.news['subcategory'] = self.news['subcategory'].map(category2int)
        
        self.news['category'] = self.news['category'].fillna(0)
        self.news['subcategory'] = self.news['subcategory'].fillna(0)

        # Store only raw fields for tokenization in __getitem__
        self.news_parsed = []
        for index, row in enumerate(self.news.itertuples()):
            text = row.title + ' ' + row.abstract
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
            max_length=config[args.dataset]['num_words_title'] + config[args.dataset]['num_words_abstract'],
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
    def __init__(self, behaviors_path):
        """
        Args:
            behaviors_path: path to behaviors file
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

        if args.cold_threshold == 0:
            max_clicked = config[args.dataset]['num_clicked_news_a_user']
        else:
            max_clicked = args.cold_threshold

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

class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path):
        """
        Args:
            behaviors_path: path to behaviors file
        """
        super(UserDataset, self).__init__()
        
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=[0, 1, 3],
                                       names=['id', 'user', 'clicked_news'])
        
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')

        user2int = dict(pd.read_table(f'data/{args.dataset}/train/user2int.tsv').values.tolist())
        
        self.behaviors['user'] = self.behaviors['user'].map(user2int)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]

        item = {
            "id": row.user,
            "clicked_news": row.clicked_news.split()[-config[args.dataset]['num_clicked_news_a_user']:]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config[args.dataset]['num_clicked_news_a_user'] - len(item["clicked_news"])

        item["clicked_news_mask"] = [0] * repeated_times + [1] * len(item["clicked_news"])
        item["clicked_news"] = ['PADDED_NEWS'] * repeated_times + item["clicked_news"]

        return item
    
class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path, config, args):
        super(BaseDataset, self).__init__()

        self.config = config
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(config[args.dataset]['pretrained_model_name'])

        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.behaviors_parsed['clicked_news'] = self.behaviors_parsed['clicked_news'].fillna(' ')

        user2int = dict(pd.read_table(f'data/{args.dataset}/train/user2int.tsv').values.tolist())
        self.behaviors_parsed['user'] = self.behaviors_parsed['user'].map(user2int)
        self.behaviors_parsed['user'] = self.behaviors_parsed['user'].fillna(0)

        self.news_parsed = pd.read_table(
            news_path,
            names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
            usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
        )

        self.news_parsed['title'] = self.news_parsed['title'].fillna('')
        self.news_parsed['abstract'] = self.news_parsed['abstract'].fillna('')

        category2int = dict(pd.read_table(f'data/{args.dataset}/train/category2int.tsv').values.tolist())
        self.news_parsed['category'] = self.news_parsed['category'].map(category2int)
        self.news_parsed['subcategory'] = self.news_parsed['subcategory'].map(category2int)
        self.news_parsed['category'] = self.news_parsed['category'].fillna(0)
        self.news_parsed['subcategory'] = self.news_parsed['subcategory'].fillna(0)

        # Store only raw fields for tokenization in __getitem__
        self.news2raw = {}
        for index, row in enumerate(self.news_parsed.itertuples()):
            text = row.title + ' ' + row.abstract
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
                max_length=self.config[self.args.dataset]['num_words_title'] + self.config[self.args.dataset]['num_words_abstract'],
                return_tensors='pt'
            )
            if (self.args.model == 'naml' or self.args.model == 'lstur') and (self.args.dataset == 'ebnerd_nyheder' or self.args.dataset == 'ebnerd_sport' or self.args.dataset == 'mind_news' or self.args.dataset == 'mind_sports'):
                return {
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'category': torch.tensor(0).long(),
                    'subcategory': torch.tensor(0).long(),
                    'text': raw['text']
                }
            else:   
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

        clicked_news_ids = row.clicked_news.split()[-self.config[self.args.dataset]['num_clicked_news_a_user']:]
        item["clicked_news"] = [
            get_tokenized(x) for x in clicked_news_ids
        ]

        item["clicked_news_length"] = len(item["clicked_news"])
        repeated_times = self.config[self.args.dataset]['num_clicked_news_a_user'] - len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news_mask"] = [0] * repeated_times + [1] * len(item["clicked_news"])
        item["clicked_news"] = [self.padding_tokenized] * repeated_times + item["clicked_news"]
        item["user"] = torch.tensor(row.user).long()
        
        return item
