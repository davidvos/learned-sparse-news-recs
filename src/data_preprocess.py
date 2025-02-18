import pandas as pd
import swifter
import json
import math
from tqdm import tqdm
from os import path
from pathlib import Path
import random
from nltk.tokenize import word_tokenize
import numpy as np
import csv
import importlib
from transformers import AutoTokenizer
from ast import literal_eval
import yaml

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

def parse_behaviors(source, target):
    """
    Parse behaviors file in training set.
    Args:
        source: source behaviors file
        target: target behaviors file
        user2int_path: path for saving user2int file
    """
    print(f"Parse {source}")

    behaviors = pd.read_table(
        source,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors.clicked_news.fillna(' ', inplace=True)

    behaviors.clicked_news = behaviors.clicked_news.swifter.apply(
            lambda x: ' '.join(x.split()))
    
    behaviors.impressions = behaviors.impressions.str.split()

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
        target,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])
    
def parse_eval_behaviors(source, target):
    """
    Parse behaviors file in evaluation set.
    Args:
        source: source behaviors file
        target: target behaviors file
    """
    print(f"Parse {source}")

    behaviors = pd.read_table(
        source,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors.clicked_news.fillna(' ', inplace=True)

    behaviors.clicked_news = behaviors.clicked_news.swifter.apply(
            lambda x: ' '.join(x.split()))
    
    behaviors.impressions = behaviors.impressions.str.split()

    behaviors['candidate_news'] = behaviors.impressions.apply(
        lambda x: ' '.join([e.split('-')[0] for e in x]))
    behaviors['clicked'] = behaviors.impressions.apply(
        lambda x: ' '.join([e.split('-')[1] for e in x]))
    
    behaviors.impressions = behaviors.impressions.apply(
        lambda x: ' '.join(x))
    
    behaviors.to_csv(
        target,
        sep='\t',
        index=False,
        header=False,
        columns=['impression_id', 'user', 'time', 'clicked_news', 'impressions', 'candidate_news', 'clicked'])

if __name__ == '__main__':

    base = './data/MIND'

    train_dir = f'{base}/train'
    val_dir = f'{base}/val'
    test_dir = f'{base}/test'

    print('Process data for training')

    print('Parse behaviors')
    parse_behaviors(path.join(train_dir, 'behaviors.tsv'),
                    path.join(train_dir, 'behaviors_parsed.tsv'))
    
    # print('Parse behaviors for validation')

    parse_eval_behaviors(path.join(val_dir, 'behaviors.tsv'),
                            path.join(val_dir, 'behaviors_parsed.tsv'))