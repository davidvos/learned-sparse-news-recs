import argparse
import yaml
import pandas as pd
from ast import literal_eval
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

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

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        if args.dataset == 'mind':
            self.news = pd.read_table(
                news_path,
                names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
            )
        else:
            self.news = pd.read_table(
                news_path,
                names=['id', 'category', 'subcategory', 'title', 'abstract', 'url'],
                usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
            )
            
        self.news['title'] = self.news['title'].fillna('')
        self.news['abstract'] = self.news['abstract'].fillna('')

        category2int = dict(pd.read_table(f'data/{args.dataset}/train/category2int.tsv').values.tolist())
        self.news['category'] = self.news['category'].map(category2int)
        self.news['subcategory'] = self.news['subcategory'].map(category2int)
        
        self.news['category'] = self.news['category'].fillna(0)
        self.news['subcategory'] = self.news['subcategory'].fillna(0)

        # print number of categories and subcategories\
        print(self.news.category)
        print(self.news.category.unique())
        print(f'Number of categories: {len(self.news.category.unique())}')
        print(f'Number of subcategories: {len(self.news.subcategory.unique())}')

        self.news_parsed = {}

        for index, row in enumerate(self.news.itertuples()):

            text = row.title + ' ' + row.abstract
            tokenized_text = tokenizer(text, padding='max_length', truncation=True, 
                                       max_length=config[args.dataset]['num_words_title'] + config[args.dataset]['num_words_abstract'], return_tensors='pt')

            self.news_parsed[index] = {}
            self.news_parsed[index]['id'] = row.id
            self.news_parsed[index]['input_ids'] = tokenized_text['input_ids']
            self.news_parsed[index]['attention_mask'] = tokenized_text['attention_mask']

            self.news_parsed[index]['category'] = torch.tensor(row.category).long()
            self.news_parsed[index]['subcategory'] = torch.tensor(row.subcategory).long()

            self.news_parsed[index]['text'] = text

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news_parsed[idx]
        return item
    
class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path, max_length=5):
        """
        Args:
            behaviors_path: path to behaviors file
            max_length: maximum length of clicked news
        """
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        self.behaviors['clicked_news_string'] = self.behaviors.clicked_news.fillna(' ')

        user2int = dict(pd.read_table(f'data/{args.dataset}/train/user2int.tsv').values.tolist())

        self.behaviors['user'] = self.behaviors['user'].map(user2int)

        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news": row.clicked_news_string.split()[:config[args.dataset]['num_clicked_news_a_user']],
            "impressions": row.impressions
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

        self.behaviors.drop_duplicates(inplace=True)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]

        item = {
            "id": row.id,
            "clicked_news": row.clicked_news.split()[:config[args.dataset]['num_clicked_news_a_user']]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config[args.dataset]['num_clicked_news_a_user'] - len(item["clicked_news"])

        item["clicked_news_mask"] = [0] * repeated_times + [1] * len(item["clicked_news"])
        item["clicked_news"] = ['PADDED_NEWS'] * repeated_times + item["clicked_news"]

        item['user'] = torch.tensor(row.user).long()

        return item
    
class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path, config, args):
        super(BaseDataset, self).__init__()

        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config[args.model]['pretrained_model_name'])

        self.behaviors_parsed = pd.read_table(behaviors_path)

        self.behaviors_parsed['clicked_news'] = self.behaviors_parsed['clicked_news'].fillna(' ')

        user2int = dict(pd.read_table(f'data/{args.dataset}/train/user2int.tsv').values.tolist())

        self.behaviors_parsed['user'] = self.behaviors_parsed['user'].map(user2int)
        self.behaviors_parsed['user'] = self.behaviors_parsed['user'].fillna(0)

        if args.dataset == 'mind':
            self.news_parsed = pd.read_table(
                news_path,
                names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
            )
        else:
            self.news_parsed = pd.read_table(
                news_path,
                names=['id', 'category', 'subcategory', 'title', 'abstract', 'url'],
                usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
            )

        self.news_parsed['title'] = self.news_parsed['title'].fillna('')
        self.news_parsed['abstract'] = self.news_parsed['abstract'].fillna('')

        category2int = dict(pd.read_table(f'data/{args.dataset}/train/category2int.tsv').values.tolist())

        self.news_parsed['category'] = self.news_parsed['category'].map(category2int)
        self.news_parsed['subcategory'] = self.news_parsed['subcategory'].map(category2int)
        
        self.news_parsed['category'] = self.news_parsed['category'].fillna(0)
        self.news_parsed['subcategory'] = self.news_parsed['subcategory'].fillna(0)

        self.news2dict = {}
        self.news2text = {}

        for index, row in enumerate(self.news_parsed.itertuples()):
            text = row.title + ' ' + row.abstract
            tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, 
                                            max_length=config[args.dataset]['num_words_title'] + config[args.dataset]['num_words_abstract'], return_tensors='pt')
            self.news2dict[str(row.id)] = {}
            self.news2dict[str(row.id)]['input_ids'] = tokenized_text['input_ids']
            self.news2dict[str(row.id)]['attention_mask'] = tokenized_text['attention_mask']

            self.news2dict[str(row.id)]['category'] = torch.tensor(row.category).long()
            self.news2dict[str(row.id)]['subcategory'] = torch.tensor(row.subcategory).long()

            self.news2text[str(row.id)] = text

        self.padding = {
            "input_ids": torch.zeros(config[args.dataset]['num_words_title'] + config[args.dataset]['num_words_abstract']).unsqueeze(0).long(),
            "attention_mask": torch.zeros(config[args.dataset]['num_words_title'] + config[args.dataset]['num_words_abstract']).unsqueeze(0).long(),
            "category": torch.tensor(0).long(),
            "subcategory": torch.tensor(0).long(),
        }
        
    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}

        row = self.behaviors_parsed.iloc[idx]

        item["candidate_news"] = [
            self.news2dict[x] for x in row.candidate_news.split()
        ]

        item["clicked_news"] = [
            self.news2dict[x]
            for x in row.clicked_news.split()[:config[args.dataset]['num_clicked_news_a_user']]
        ]

        item["clicked_news_length"] = len(item["clicked_news"])
        
        repeated_times = config[args.dataset]['num_clicked_news_a_user'] - len(item["clicked_news"])
        assert repeated_times >= 0
        
        item["clicked_news_mask"] = [0] * repeated_times + [1] * len(item["clicked_news"])
        item["clicked_news"] = [self.padding] * repeated_times + item["clicked_news"]

        item["user"] = torch.tensor(row.user).long()

        return item
