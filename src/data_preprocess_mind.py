"""
Preprocessing script for the MIND news recommendation dataset.

This script parses and processes the behaviors files for training, validation, and test splits.
It performs negative sampling for training, and supports cold-start filtering and user-based grouping for evaluation splits.

Outputs are saved as tab-separated files ready for model training and evaluation.
"""
import pandas as pd
from tqdm import tqdm
from os import path
import random
import yaml
import os

# Load configuration
with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

def parse_behaviors(source_path, target_path):
    """
    Parse and balance the training behaviors file with negative sampling.

    Args:
        source_path (str): Path to the source behaviors.tsv file.
        target_path (str): Path to save the processed output file.
    """
    behaviors = pd.read_table(
        source_path,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    # Fill missing clicked_news with empty string
    behaviors['clicked_news'] = behaviors['clicked_news'].fillna(' ')

    # Normalize whitespace in clicked_news
    behaviors.clicked_news = behaviors.clicked_news.apply(
        lambda x: ' '.join(x.split()))
    
    # Split impressions into list
    behaviors.impressions = behaviors.impressions.str.split()

    # For each row, create balanced positive/negative impression pairs
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

    # Explode pairs into separate rows
    behaviors = behaviors.explode('impressions').dropna(
        subset=["impressions"]).reset_index(drop=True)
    # Split impression pairs into candidate_news and clicked labels
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (' '.join([e.split('-')[0] for e in x]), ' '.join(
                [e.split('-')[1] for e in x]))).tolist())
    
    # Save processed file
    behaviors.to_csv(
        target_path,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])
    
def parse_eval_behaviors(source_path, target_path, cold_threshold, split='val'):
    """
    Parse and group the evaluation (val/test) behaviors file, with optional cold-start filtering.

    Args:
        source_path (str): Path to the source behaviors.tsv file.
        target_path (str): Path to save the processed output file.
        cold_threshold (int): Max number of clicked_news for a user to be included (0 disables filtering).
        split (str): 'val' or 'test' to determine which half of users to use.
    """
    behaviors = pd.read_table(
        source_path,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors['clicked_news'] = behaviors['clicked_news'].fillna(' ')

    # Filter for cold-start users if threshold > 0
    if cold_threshold != 0:
        behaviors = behaviors[behaviors.clicked_news.apply(lambda x: len(x.split())) <= cold_threshold]
        behaviors = behaviors[behaviors.clicked_news.apply(lambda x: len(x.split())) > 0]
    
    # Split impressions into list
    behaviors.impressions = behaviors.impressions.str.split()

    # Extract candidate news and clicked labels for each impression
    behaviors['candidate_news'] = behaviors.impressions.apply(
        lambda x: ' '.join([e.split('-')[0] for e in x]))
    behaviors['clicked'] = behaviors.impressions.apply(
        lambda x: ' '.join([e.split('-')[1] for e in x]))
    
    # Join impressions back into a single string for grouping
    behaviors.impressions = behaviors.impressions.apply(
        lambda x: ' '.join(x))
    
    # Group by user and combine all impressions for each user
    print("Grouping impressions by user...")
    grouped_behaviors = behaviors.groupby('user').agg({
        'impression_id': lambda x: ' '.join(map(str, x)),
        'time': lambda x: ' '.join(map(str, x)),
        'clicked_news': lambda x: x.iloc[0],  # Always take the first clicked_news
        'impressions': lambda x: ' '.join(x),
        'candidate_news': lambda x: ' '.join(x),
        'clicked': lambda x: ' '.join(x)
    }).reset_index()

    # Split users into val/test halves
    if split == 'val':
        grouped_behaviors = grouped_behaviors.iloc[:int(len(grouped_behaviors) * 0.5)].reset_index(drop=True)
    elif split == 'test':
        grouped_behaviors = grouped_behaviors.iloc[int(len(grouped_behaviors) * 0.5):].reset_index(drop=True)
    
    # Save processed file
    grouped_behaviors.to_csv(
        target_path,
        sep='\t',
        index=False,
        header=False,
        columns=['impression_id', 'user', 'time', 'clicked_news', 'impressions', 'candidate_news', 'clicked'])


def create_user2int_category2int(behaviors_paths, news_paths, output_dirs):
    """
    Create user2int.tsv and category2int.tsv files from all splits (train, val, test),
    and write them to each split's directory.
    Args:
        behaviors_paths: list of paths to behaviors.tsv files (train, val, test)
        news_paths: list of paths to news.tsv files (train, val, test)
        output_dirs: list of directories to save the mapping files (train, val, test)
    """
    # User2int
    all_users = set()
    for p in behaviors_paths:
        for chunk in pd.read_table(p, header=None, usecols=[1], names=['user'], chunksize=100000):
            all_users.update(chunk['user'].unique())
    user2int = {user: idx+1 for idx, user in enumerate(sorted(all_users))}
    user2int_df = pd.DataFrame(list(user2int.items()), columns=['user','int'])

    # Category2int
    all_categories = set()
    for p in news_paths:
        for chunk in pd.read_table(p, header=None, usecols=[1,2], names=['category','subcategory'], chunksize=100000):
            all_categories.update(chunk['category'].dropna().unique())
            all_categories.update(chunk['subcategory'].dropna().unique())
    all_categories = [c for c in all_categories if pd.notnull(c)]
    category2int = {cat: idx+1 for idx, cat in enumerate(sorted(all_categories))}
    category2int_df = pd.DataFrame(list(category2int.items()), columns=['category','int'])

    # Write to each output directory
    for out_dir in output_dirs:
        user2int_path = os.path.join(out_dir, 'user2int.tsv')
        category2int_path = os.path.join(out_dir, 'category2int.tsv')
        user2int_df.to_csv(user2int_path, sep='\t', index=False, header=False)
        category2int_df.to_csv(category2int_path, sep='\t', index=False, header=False)


if __name__ == '__main__':
    """
    Main entry point for preprocessing MIND dataset behaviors files.
    Processes training, validation, and test splits with various cold-start thresholds.
    """
    base = './data/mind'
    train_dir = f'{base}/train'
    val_dir = f'{base}/val'
    test_dir = f'{base}/test'

    # Create user2int and category2int mappings
    create_user2int_category2int(
        [path.join(train_dir, 'behaviors.tsv'), path.join(val_dir, 'behaviors.tsv'), path.join(test_dir, 'behaviors.tsv')],
        [path.join(train_dir, 'news.tsv'), path.join(val_dir, 'news.tsv'), path.join(test_dir, 'news.tsv')],
        [train_dir, val_dir, test_dir]
    )

    print('Process data for training')
    print('Parse behaviors')
    parse_behaviors(path.join(train_dir, 'behaviors.tsv'),
                    path.join(train_dir, f'behaviors_parsed.tsv'))
    
    for cold_threshold in [0, 1, 2, 3, 4, 5]:
        print(f'Parse validation behaviors with cold threshold {cold_threshold}')
        parse_eval_behaviors(
            path.join(val_dir, 'behaviors.tsv'),
            path.join(val_dir, f'behaviors_parsed_cold{cold_threshold}.tsv'),
            cold_threshold=cold_threshold,
            split='val'
        )

        print(f'Parse test behaviors with cold threshold {cold_threshold}')
        parse_eval_behaviors(
            path.join(test_dir, 'behaviors.tsv'),
            path.join(test_dir, f'behaviors_parsed_cold{cold_threshold}.tsv'),
            cold_threshold=cold_threshold,
            split='test'
        )