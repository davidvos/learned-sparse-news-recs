# -*- coding: utf-8 -*-
"""
Data preprocessing for MIND OOD (Out-Of-Distribution) experiments.
This script filters and processes user behavior and news data for OOD evaluation by category.
"""
import pandas as pd
import random
import yaml
from os import path
from tqdm import tqdm

# Load configuration from YAML file
with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Removed filter_out_category and filter_in_category functions

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
        user2int_path = path.join(out_dir, 'user2int.tsv')
        category2int_path = path.join(out_dir, 'category2int.tsv')
        user2int_df.to_csv(user2int_path, sep='\t', index=False, header=False)
        category2int_df.to_csv(category2int_path, sep='\t', index=False, header=False)


def parse_train_behaviors(source_path, target_path, news_path, target_category):
    """
    Parse and filter training behaviors:
    - Remove all news of the target category from histories and candidates.
    - Remove samples with no history or candidates left.
    - Perform negative sampling and construct candidate pairs.
    - Save the processed behaviors to a TSV file.
    """
    # Load news data
    news = pd.read_table(
        news_path,
        names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
        usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
    )
    # Only keep news not in the target category 
    allowed_news_ids = set(news[news['category'] != target_category]['id'])

    # Load behaviors data
    behaviors = pd.read_table(
        source_path,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors['clicked_news'] = behaviors['clicked_news'].fillna(' ')

    # Remove target category news from histories 
    behaviors.clicked_news = behaviors.clicked_news.apply(
        lambda x: ' '.join([e for e in x.split() if e in allowed_news_ids]))

    # Remove target category news from candidates 
    def filter_candidates(impr):
        return ' '.join([e for e in impr.split() if e.split('-')[0] in allowed_news_ids])
    behaviors.impressions = behaviors.impressions.apply(filter_candidates)

    # Remove samples with no history or no candidates
    behaviors = behaviors[(behaviors['clicked_news'].str.strip() != '') & (behaviors['impressions'].str.strip() != '')]

    # Only keep samples with at least one positive candidate
    behaviors = behaviors[behaviors['impressions'].str.split().apply(lambda x: any([e.endswith('1') for e in x]))]

    # Split impressions into lists
    behaviors['impressions'] = behaviors['impressions'].str.split()

    # Negative sampling and pair construction
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
    behaviors = behaviors.explode('impressions').dropna(subset=["impressions"]).reset_index(drop=True)

    # Split impressions into candidate_news and clicked labels
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (' '.join([e.split('-')[0] for e in x]), ' '.join([e.split('-')[1] for e in x]))).tolist())

    # Save processed behaviors
    behaviors.to_csv(
        target_path,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])


def parse_eval_behaviors(source_path, val_target_path, test_target_path, news_path, target_category):
    """
    Parse and filter evaluation behaviors:
    - Only keep impressions where candidate news is of the target category.
    - Remove samples with no history or candidates left.
    - Group by user, aggregate impressions, and split into validation and test sets (first half for val, second half for test).
    - Save the processed behaviors to TSV files for validation and test.
    """
    # Load news data
    news = pd.read_table(
        news_path,
        names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
        usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
    )
    # Only keep news in the target category 
    target_news_ids = set(news[news['category'] == target_category]['id'])

    # Load behaviors data
    behaviors = pd.read_table(
        source_path,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors['clicked_news'] = behaviors['clicked_news'].fillna(' ')

    # Remove non-target category news from candidates 
    def filter_candidates(impr):
        return ' '.join([e for e in impr.split() if e.split('-')[0] in target_news_ids])
    behaviors.impressions = behaviors.impressions.apply(filter_candidates)

    # Remove samples with no history or no candidates
    behaviors = behaviors[(behaviors['clicked_news'].str.strip() != '') & (behaviors['impressions'].str.strip() != '')]

    # Only keep samples with at least one positive candidate
    behaviors = behaviors[behaviors['impressions'].str.split().apply(lambda x: any([e.endswith('1') for e in x]))]

    # Split impressions into lists
    behaviors.impressions = behaviors.impressions.str.split()

    # Extract candidate_news and clicked labels
    behaviors['candidate_news'] = behaviors.impressions.apply(lambda x: ' '.join([e.split('-')[0] for e in x]))
    behaviors['clicked'] = behaviors.impressions.apply(lambda x: ' '.join([e.split('-')[1] for e in x]))
    behaviors.impressions = behaviors.impressions.apply(lambda x: ' '.join(x))

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

    # Split into validation and test sets
    n = len(grouped_behaviors)
    val_behaviors = grouped_behaviors.iloc[:n//2].reset_index(drop=True)
    test_behaviors = grouped_behaviors.iloc[n//2:].reset_index(drop=True)

    # Save validation and test behaviors
    val_behaviors.to_csv(
        val_target_path,
        sep='\t',
        index=False,
        header=False,
        columns=['impression_id', 'user', 'time', 'clicked_news', 'impressions', 'candidate_news', 'clicked'])
    test_behaviors.to_csv(
        test_target_path,
        sep='\t',
        index=False,
        header=False,
        columns=['impression_id', 'user', 'time', 'clicked_news', 'impressions', 'candidate_news', 'clicked'])


if __name__ == '__main__':
    # Set the target category for OOD split
    target_category = 'news'  # Change as needed

    # Define data directories
    base = f'./data/mind_{target_category}'
    train_dir = f'{base}/train'
    val_dir = f'{base}/val'
    test_dir = f'{base}/test'

    # Create user2int and category2int mappings
    create_user2int_category2int(
        [path.join(train_dir, 'behaviors.tsv'), path.join(val_dir, 'behaviors.tsv'), path.join(test_dir, 'behaviors.tsv')],
        [path.join(train_dir, 'news.tsv'), path.join(val_dir, 'news.tsv'), path.join(test_dir, 'news.tsv')],
        [train_dir, val_dir, test_dir]
    )

    # Process training behaviors
    parse_train_behaviors(
        path.join(train_dir, 'behaviors.tsv'),
        path.join(train_dir, 'behaviors_parsed.tsv'),
        path.join(train_dir, 'news.tsv'),
        target_category
    )

    # Process evaluation (validation and test) behaviors
    parse_eval_behaviors(
        path.join(val_dir, 'behaviors.tsv'),
        path.join(val_dir, 'behaviors_parsed_cold0.tsv'),
        path.join(test_dir, 'behaviors_parsed_cold0.tsv'),
        path.join(train_dir, 'news.tsv'),
        target_category
    )