"""
Preprocessing script for the ebnerd news recommendation dataset.

This script parses and processes the behaviors and news files for training, validation, and test splits.
It performs negative sampling for training, supports cold-start filtering for evaluation splits, and generates user/category mappings.

Outputs are saved as tab-separated files ready for model training and evaluation.
"""
import pandas as pd
from tqdm import tqdm
import random
import yaml
from os import path
import os

# Load configuration from YAML
with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

def create_impressions(candidates, labels):
    """
    Create a list of 'newsid-label' strings from candidates and labels.
    Args:
        candidates: list of candidate news ids
        labels: list of clicked news ids
    Returns:
        String of space-separated 'newsid-label' pairs (e.g., 'A-1 B-0 C-0')
    """
    impression_list = [f"{nid}-{'1' if nid in labels else '0'}" for nid in candidates]
    impressions = ' '.join(impression_list)
    return impressions

def extract_first(x):
    """
    Extract the first element from a list/array, or return the value itself if not a list.
    Returns empty string for nulls or empty lists.
    """
    if isinstance(x, (list, tuple)) or (hasattr(x, 'dtype') and hasattr(x, '__getitem__')):
        if len(x) > 0:
            return x[0]
        else:
            return ''
    elif pd.isnull(x):
        return ''
    else:
        return x

def parse_behaviors(behaviors_path, history_path, target_path):
    """
    Parse and balance the training behaviors file with negative sampling.
    Args:
        behaviors_path: Path to behaviors.parquet
        history_path: Path to history.parquet
        target_path: Output path for processed behaviors .tsv
    """
    # Load behaviors and histories, merge on user_id
    behaviors = pd.read_parquet(behaviors_path)
    histories = pd.read_parquet(history_path)
    behaviors = behaviors.merge(histories, on='user_id', how='left')

    # Create impressions string (e.g., 'A-1 B-0 C-0')
    behaviors['impressions'] = behaviors.apply(
        lambda x: create_impressions(x['article_ids_inview'], x['article_ids_clicked']),
        axis=1
    )
    behaviors['impressions'] = behaviors['impressions'].str.split()

    # Create clicked_news string
    behaviors['clicked_news'] = behaviors['article_id_fixed'].apply(lambda x: ' '.join(map(str, list(x))))

    # Keep only relevant columns
    behaviors = behaviors[['impression_id', 'user_id', 'impression_time', 'clicked_news', 'impressions']]

    # Negative sampling: for each row, create balanced positive/negative impression pairs
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

    # Split impression pairs into candidate_news and clicked labels
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (' '.join([e.split('-')[0] for e in x]), ' '.join([e.split('-')[1] for e in x]))).tolist())

    # Rename columns for output
    behaviors = behaviors.rename(columns={'user_id': 'user'})

    # Save processed file (tab-separated, no header)
    behaviors.to_csv(target_path, sep='\t', index=False, header=False, columns=['user', 'clicked_news', 'candidate_news', 'clicked'])

def parse_val_behaviors(behaviors_path, history_path, target_path, cold_threshold, split='val'):
    """
    Parse and group the evaluation (val/test) behaviors file, with optional cold-start filtering.
    Args:
        behaviors_path: Path to behaviors.parquet
        history_path: Path to history.parquet
        target_path: Output path for processed eval behaviors .tsv
        cold_threshold: Max number of clicked_news for a user to be included (0 disables filtering)
        split: 'val' or 'test' to determine which half of users to use
    """
    # Load and merge behaviors and histories
    behaviors = pd.read_parquet(behaviors_path)
    histories = pd.read_parquet(history_path)
    behaviors = behaviors.merge(histories, on='user_id', how='left')

    # Create impressions string
    behaviors['impressions'] = behaviors.apply(
        lambda x: create_impressions(x['article_ids_inview'], x['article_ids_clicked']),
        axis=1
    )

    # Create clicked_news string and apply cold-start filtering
    behaviors['clicked_news'] = behaviors['article_id_fixed'].apply(lambda x: ' '.join(map(str, list(x))))
    behaviors['clicked_news'] = behaviors['clicked_news'].apply(lambda x: x.split(' ')[-cold_threshold:])
    behaviors['clicked_news'] = behaviors['clicked_news'].apply(lambda x: ' '.join(map(str, x)))

    # Group by user_id and aggregate as space-separated strings
    grouped = behaviors.groupby('user_id').agg({
        'impression_id': lambda x: ' '.join(map(str, x)),
        'impression_time': lambda x: ' '.join(map(str, x)),
        'clicked_news': lambda x: x.iloc[0],
        'impressions': lambda x: ' '.join(x)
    }).reset_index()

    # Split users into val/test halves
    if split == 'val':
        grouped = grouped.iloc[:int(len(grouped) * 0.5)].reset_index(drop=True)
    elif split == 'test':
        grouped = grouped.iloc[int(len(grouped) * 0.5):].reset_index(drop=True)

    # Save processed file (tab-separated, no header)
    grouped.to_csv(target_path, sep='\t', index=False, header=False, columns=['impression_id', 'user_id', 'impression_time', 'clicked_news', 'impressions'])

def parse_news(news_path, target_path):
    """
    Parse news file to align with MIND format.
    Args:
        news_path: Path to articles.parquet
        target_path: Output path for processed news .tsv
    """
    news = pd.read_parquet(news_path)

    # Extract first subcategory if list-like
    news['subcategory'] = news['subcategory'].apply(extract_first)

    # Clean title and body text
    for col in ['title', 'subtitle', 'body']:
        news[col] = news[col].astype(str).str.replace(r'\n', ' ', regex=True)
        news[col] = news[col].str.replace(r'\t', ' ', regex=True)
        news[col] = news[col].str.replace(r'\s+', ' ', regex=True)
        news[col] = news[col].str.strip()

    # Combine title and subtitle
    news['title'] = news['title'].astype(str) + ' ' + news['subtitle'].astype(str)
    news = news.drop(columns=['subtitle'])

    # Add empty entity columns for compatibility
    news['title_entities'] = '[]'
    news['abstract_entities'] = '[]'

    # Rename columns to match MIND
    news = news.rename(columns={'article_id': 'id', 'body': 'abstract'})

    # Select and order columns as in MIND
    news = news[['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']]

    # Save processed file (tab-separated, no header)
    news.to_csv(target_path, sep='\t', index=False, header=False)

def create_user2int_category2int(behaviors_paths, news_paths, output_dirs):
    """
    Create user2int.tsv and category2int.tsv files from all splits (train, val, test),
    and write them to each split's directory.
    Args:
        behaviors_paths: List of paths to behaviors.parquet files (train, val, test)
        news_paths: List of paths to articles.parquet files (train, val, test)
        output_dirs: List of directories to save the mapping files (train, val, test)
    """
    # Aggregate all users from all splits
    all_behaviors = [pd.read_parquet(p) for p in behaviors_paths]
    all_users = pd.concat([df['user_id'] for df in all_behaviors]).unique()
    user2int = {user: idx+1 for idx, user in enumerate(all_users)}
    user2int_df = pd.DataFrame(list(user2int.items()), columns=['user','int'])

    # Aggregate all categories and subcategories from all splits
    all_news = [pd.read_parquet(p) for p in news_paths]
    for df in all_news:
        df['category'] = df['category'].apply(extract_first)
        df['subcategory'] = df['subcategory'].apply(extract_first)
    all_categories = pd.concat([pd.concat([df['category'], df['subcategory']]) for df in all_news]).unique()
    all_categories = [c for c in all_categories if pd.notnull(c)]
    category2int = {cat: idx+1 for idx, cat in enumerate(all_categories)}
    category2int_df = pd.DataFrame(list(category2int.items()), columns=['category','int'])

    # Write mapping files to each output directory
    for out_dir in output_dirs:
        user2int_path = os.path.join(out_dir, 'user2int.tsv')
        category2int_path = os.path.join(out_dir, 'category2int.tsv')
        user2int_df.to_csv(user2int_path, sep='\t', index=False, header=False)
        category2int_df.to_csv(category2int_path, sep='\t', index=False, header=False)

if __name__ == '__main__':
    # Main entry point for preprocessing ebnerd dataset behaviors files.
    # Processes training, validation, and test splits with various cold-start thresholds.
    base = './data/ebnerd'
    train_dir = f'{base}/train'
    val_dir = f'{base}/val'
    test_dir = f'{base}/test'

    # Create user2int and category2int mappings
    create_user2int_category2int(
        [path.join(train_dir, 'behaviors.parquet'), path.join(val_dir, 'behaviors.parquet'), path.join(test_dir, 'behaviors.parquet')],
        [path.join(base, 'articles.parquet')],
        [train_dir, val_dir, test_dir]
    )

    # Process training behaviors
    parse_behaviors(path.join(train_dir, 'behaviors.parquet'),
                    path.join(train_dir, f'history.parquet'),
                    path.join(train_dir, f'behaviors_parsed.tsv'))

    # Process validation and test splits for various cold thresholds
    for cold_threshold in [1, 2, 3, 4, 5, 0]:
        parse_val_behaviors(path.join(val_dir, 'behaviors.parquet'),
                        path.join(val_dir, f'history.parquet'),
                        path.join(val_dir, f'behaviors_parsed_cold{cold_threshold}.tsv'),
                        cold_threshold,
                        split='val')
        parse_val_behaviors(path.join(test_dir, 'behaviors.parquet'),
                        path.join(test_dir, f'history.parquet'),
                        path.join(test_dir, f'behaviors_parsed_cold{cold_threshold}.tsv'),
                        cold_threshold,
                        split='test')

    # Process news files for each split
    parse_news(path.join(base, 'articles.parquet'),
               path.join(train_dir, f'news.tsv'))
    parse_news(path.join(base, 'articles.parquet'),
               path.join(val_dir, f'news.tsv'))
    parse_news(path.join(base, 'articles.parquet'),
               path.join(test_dir, f'news.tsv')) 