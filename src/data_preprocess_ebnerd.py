from os import path
import pandas as pd
from tqdm import tqdm
import random
import yaml

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

def create_impressions(candidates, labels):
    """
    Create a list of 'newsid-label' strings from candidates and labels.
    Args:
        candidates: list of candidate news ids
        labels: list of clicked news ids
    Returns:
        List of 'newsid-label' strings
    """
    impression_list = [f"{nid}-{'1' if nid in labels else '0'}" for nid in candidates]
    impressions = ' '.join(impression_list)
    return impressions

def extract_first(x):
        # If x is a list or array-like
        if isinstance(x, (list, tuple)) or (hasattr(x, 'dtype') and hasattr(x, '__getitem__')):
            if len(x) > 0:
                return x[0]
            else:
                return ''
        # If x is a scalar null
        elif pd.isnull(x):
            return ''
        else:
            return x

def parse_behaviors(behaviors_path, history_path, target_path):
    """
    Parse behaviors file in training set (aligned with MIND logic).
    Args:
        behaviors_path: source behaviors file
        history_path: source history file
        target_path: target behaviors file
    """
    behaviors = pd.read_parquet(behaviors_path)
    histories = pd.read_parquet(history_path)
    behaviors = behaviors.merge(histories, on='user_id', how='left')

    behaviors['impressions'] = behaviors.apply(
        lambda x: create_impressions(x['article_ids_inview'], x['article_ids_clicked']),
        axis=1
    )

    behaviors.impressions = behaviors.impressions.str.split()
    
    behaviors['clicked_news'] = behaviors['article_id_fixed'].apply(lambda x: ' '.join(map(str, list(x))))

    behaviors = behaviors[['impression_id', 'user_id', 'impression_time', 'clicked_news', 'impressions']]

    # Balance (negative sampling) and explode
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

    behaviors = behaviors.explode('impressions').dropna(subset=["impressions"]).reset_index(drop=True)

    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (' '.join([e.split('-')[0] for e in x]), ' '.join([e.split('-')[1] for e in x]))).tolist())

    print(behaviors.head())
    # Rename user_id to user
    behaviors = behaviors.rename(columns={'user_id': 'user'})

    # Output columns: user, clicked_news, candidate_news, clicked (tab-separated, no header)
    behaviors.to_csv(target_path, sep='\t', index=False, header=True, columns=['user', 'clicked_news', 'candidate_news', 'clicked'])

def parse_val_behaviors(behaviors_path, history_path, target_path, cold_threshold, split='val'):
    """
    Parse behaviors file in evaluation set (aligned with MIND logic).
    Args:
        behaviors_path: source behaviors file
        history_path: source history file
        target_path: target behaviors file
        cold_threshold: threshold for cold start (applied to clicked_news)
    """
    behaviors = pd.read_parquet(behaviors_path)
    histories = pd.read_parquet(history_path)
    behaviors = behaviors.merge(histories, on='user_id', how='left')
   

    behaviors['impressions'] = behaviors.apply(
        lambda x: create_impressions(x['article_ids_inview'], x['article_ids_clicked']),
        axis=1
    )
    
    behaviors['clicked_news'] = behaviors['article_id_fixed'].apply(lambda x: ' '.join(map(str, list(x))))

    behaviors['clicked_news'] = behaviors['clicked_news'].apply(lambda x: x.split(' ')[-cold_threshold:])
    behaviors['clicked_news'] = behaviors['clicked_news'].apply(lambda x: ' '.join(map(str, x)))
        
    # Group by user_id and aggregate as space-separated strings
    grouped = behaviors.groupby('user_id').agg({
        'impression_id': lambda x: ' '.join(map(str, x)),
        'impression_time': lambda x: ' '.join(map(str, x)),
        'clicked_news': lambda x: x.iloc[0],  # or ' '.join(x) if you want all
        'impressions': lambda x: ' '.join(x)
    }).reset_index()

    if split == 'val':
        grouped = grouped.iloc[:int(len(grouped) * 0.5)].reset_index(drop=True)
    elif split == 'test':
        grouped = grouped.iloc[int(len(grouped) * 0.5):].reset_index(drop=True)

    # Output columns: user_id, impression_id, impression_time, clicked_news, impressions (tab-separated, no header)
    grouped.to_csv(target_path, sep='\t', index=False, header=False, columns=['impression_id', 'user_id', 'impression_time', 'clicked_news', 'impressions'])

def parse_news(news_path, target_path):
    """
    Parse news file to align with MIND format.
    Args:
        news_path: source news file
        target_path: target news file
    """
    news = pd.read_parquet(news_path)
    # Robust subcategory extraction


    news['subcategory'] = news['subcategory'].apply(extract_first)

    # Clean title and body
    for col in ['title', 'subtitle', 'body']:
        news[col] = news[col].astype(str).str.replace(r'\n', ' ', regex=True)
        news[col] = news[col].str.replace(r'\t', ' ', regex=True)
        news[col] = news[col].str.replace(r'\s+', ' ', regex=True)
        news[col] = news[col].str.strip()

    # combine title and subtitle
    news['title'] = news['title'].astype(str) + ' ' + news['subtitle'].astype(str)
    news = news.drop(columns=['subtitle'])

    # Add empty entity columns
    news['title_entities'] = '[]'
    news['abstract_entities'] = '[]'

    # Rename columns to match MIND
    news = news.rename(columns={'article_id': 'id', 'body': 'abstract'})

    # Select and order columns as in MIND
    news = news[['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']]

    news.to_csv(target_path, sep='\t', index=False, header=False)

def create_user2int_category2int(behaviors_paths, news_paths, output_dirs):
    """
    Create user2int.tsv and category2int.tsv files from all splits (train, val, test),
    and write them to each split's directory.
    Args:
        behaviors_paths: list of paths to behaviors.parquet files (train, val, test)
        news_paths: list of paths to articles.parquet files (train, val, test)
        output_dirs: list of directories to save the mapping files (train, val, test)
    """
    import os
    # User2int
    all_behaviors = [pd.read_parquet(p) for p in behaviors_paths]
    all_users = pd.concat([df['user_id'] for df in all_behaviors]).unique()
    user2int = {user: idx+1 for idx, user in enumerate(all_users)}
    user2int_df = pd.DataFrame(list(user2int.items()), columns=['user','int'])

    # Category2int
    all_news = [pd.read_parquet(p) for p in news_paths]
    # Apply extract_first to category and subcategory columns
    for df in all_news:
        df['category'] = df['category'].apply(extract_first)
        df['subcategory'] = df['subcategory'].apply(extract_first)
    all_categories = pd.concat([pd.concat([df['category'], df['subcategory']]) for df in all_news]).unique()
    all_categories = [c for c in all_categories if pd.notnull(c)]
    category2int = {cat: idx+1 for idx, cat in enumerate(all_categories)}
    category2int_df = pd.DataFrame(list(category2int.items()), columns=['category','int'])

    # Write to each output directory
    for out_dir in output_dirs:
        user2int_path = os.path.join(out_dir, 'user2int.tsv')
        category2int_path = os.path.join(out_dir, 'category2int.tsv')
        user2int_df.to_csv(user2int_path, sep='\t', index=False, header=False)
        category2int_df.to_csv(category2int_path, sep='\t', index=False, header=False)

if __name__ == '__main__':

    base = './data/ebnerd'
    
    train_dir = f'{base}/train'
    val_dir = f'{base}/val'
    test_dir = f'{base}/test'

    # create_user2int_category2int(
    #     [path.join(train_dir, 'behaviors.parquet'), path.join(val_dir, 'behaviors.parquet'), path.join(test_dir, 'behaviors.parquet')],
    #     [path.join(train_dir, 'articles.parquet'), path.join(val_dir, 'articles.parquet'), path.join(test_dir, 'articles.parquet')],
    #     [train_dir, val_dir, test_dir]
    # )

    # parse_behaviors(path.join(train_dir, 'behaviors.parquet'),
    #                 path.join(train_dir, f'history.parquet'),
    #                 path.join(train_dir, f'behaviors_parsed.tsv'))

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

    parse_news(path.join(train_dir, 'articles.parquet'),
               path.join(train_dir, f'news.tsv'))
    parse_news(path.join(val_dir, 'articles.parquet'),
               path.join(val_dir, f'news.tsv'))
    parse_news(path.join(test_dir, 'articles.parquet'),
               path.join(test_dir, f'news.tsv')) 