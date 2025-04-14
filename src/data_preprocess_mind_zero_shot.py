import pandas as pd
import random
import yaml

from os import path
from tqdm import tqdm

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

def parse_behaviors(source_path, target_path, news_path, source_category):
    """
    Parse behaviors file in training set.
    Args:
        source_file: source behaviors file
        target_file: target behaviors file
        news_path: path for news file
        source_category: source category
    """
    news = pd.read_table(
        news_path,
        names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
        usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
    )

    source_category_news_ids = news[news['category'] == source_category]['id'].tolist()

    behaviors = pd.read_table(
        source_path,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors.clicked_news.fillna(' ', inplace=True)

    behaviors.clicked_news = behaviors.clicked_news.apply(
        lambda x: ' '.join([e for e in x.split() if e in source_category_news_ids]))
    
    behaviors.impressions = behaviors.impressions.apply(
        lambda x: ' '.join([e for e in x.split() if e.split('-')[0] in source_category_news_ids]))
    
    behaviors = behaviors[behaviors['impressions'].str.strip() != '']

    behaviors = behaviors[behaviors['impressions'].str.split().apply(
        lambda x: any([e.endswith('1') for e in x]))]
        
    behaviors['impressions'] = behaviors['impressions'].str.split()

    for row in tqdm(behaviors.itertuples(), desc="Balancing data"):
        positive = iter([x for x in row.impressions if x.endswith('1')])
        negative = [x for x in row.impressions if x.endswith('0')]
        random.shuffle(negative)
        negative = iter(negative)
        pairs = []
        try:
            while True:
                pair = [next(positive)]
                for _ in range(config['negative_sampling_ratio']):
                    pair.append(next(negative))
                pairs.append(pair)
        except StopIteration:
            pass
        behaviors.at[row.Index, 'impressions'] = pairs

    behaviors = behaviors.explode('impressions').dropna(
        subset=["impressions"]).reset_index(drop=True)
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (' '.join([e.split('-')[0] for e in x]), ' '.join(
                [e.split('-')[1] for e in x]))).tolist())
    
    behaviors.to_csv(
        target_path,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])
    
def parse_eval_behaviors(source_path, target_path, news_path, target_category):
    """
    Parse behaviors file in evaluation set.
    Args:
        source_file: source behaviors file
        target_file: target behaviors file
        news_path: path for news file
        target_category: target category
    """
    behaviors = pd.read_table(
        source_path,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors.clicked_news.fillna(' ', inplace=True)

    news = pd.read_table(
        news_path,
        names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
        usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
    )

    target_category_news_ids = news[news['category'] == target_category]['id'].tolist()

    behaviors.clicked_news = behaviors.clicked_news.apply(
        lambda x: ' '.join([e for e in x.split() if e in target_category_news_ids]))
    
    behaviors.impressions = behaviors.impressions.apply(
        lambda x: ' '.join([e for e in x.split() if e.split('-')[0] in target_category_news_ids]))
    
    behaviors = behaviors[behaviors['impressions'].str.strip() != '']

    behaviors = behaviors[behaviors['impressions'].str.split().apply(
        lambda x: any([e.endswith('1') for e in x]))]
    
    behaviors.impressions = behaviors.impressions.str.split()

    behaviors['candidate_news'] = behaviors.impressions.apply(
        lambda x: ' '.join([e.split('-')[0] for e in x]))
    behaviors['clicked'] = behaviors.impressions.apply(
        lambda x: ' '.join([e.split('-')[1] for e in x]))
    
    behaviors.impressions = behaviors.impressions.apply(
        lambda x: ' '.join(x))
    
    behaviors.to_csv(
        target_path,
        sep='\t',
        index=False,
        header=False,
        columns=['impression_id', 'user', 'time', 'clicked_news', 'impressions', 'candidate_news', 'clicked'])

if __name__ == '__main__':

    source_category = 'news'
    target_category = 'sports'

    base = f'./data/mind_zero_shot_{source_category}_{target_category}'

    train_dir = f'{base}/train'
    val_dir = f'{base}/val'
    test_dir = f'{base}/test'
    
    parse_behaviors(path.join(train_dir, 'behaviors.tsv'),
                    path.join(train_dir, f'behaviors_parsed.tsv'),
                    path.join(train_dir, 'news.tsv'),
                    source_category)
    
    
    parse_eval_behaviors(path.join(val_dir, 'behaviors.tsv'),
                    path.join(val_dir, f'behaviors_parsed_0.tsv'),
                    path.join(train_dir, 'news.tsv'),
                    target_category)