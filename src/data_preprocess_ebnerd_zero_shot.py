import pandas as pd
import random
import yaml

from os import path
from tqdm import tqdm


with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

def parse_behaviors(behaviors_path, history_path, target_path, articles_path, target_category):
    """
    Parse behaviors file in training set.
    Remove all news with the target category from histories and candidates.
    Args:
        behaviors_path: source behaviors file
        history_path: source history file
        target_path: target behaviors file
        articles_path: path to articles parquet
        target_category: category to filter out
    """
    behaviors = pd.read_parquet(behaviors_path)
    histories = pd.read_parquet(history_path)
    behaviors = behaviors.merge(histories, on='user_id', how='left')
    behaviors = behaviors[['impression_id', 'user_id', 'impression_time', 'article_id_fixed', 
                           'article_ids_inview', 'article_ids_clicked']]
    articles = pd.read_parquet(articles_path)
    # Only keep news NOT in the target category
    allowed_news_ids = set(articles[articles['category_str'] != target_category]['article_id'])
    behaviors['article_id_fixed'] = behaviors['article_id_fixed'].apply(
        lambda x: [e for e in x if e in allowed_news_ids])
    behaviors['article_ids_inview'] = behaviors['article_ids_inview'].apply(
        lambda x: [e for e in x if e in allowed_news_ids])
    behaviors['article_ids_clicked'] = behaviors['article_ids_clicked'].apply(
        lambda x: [e for e in x if e in allowed_news_ids])
    behaviors = behaviors[behaviors['article_id_fixed'].apply(len) > 0]
    behaviors = behaviors[behaviors['article_ids_inview'].apply(len) > 0]
    behaviors = behaviors[behaviors['article_ids_clicked'].apply(len) > 0]
    behaviors['clicked'] = behaviors.apply(lambda x: ' '.join(['1' if i in x['article_ids_clicked'] else '0' for i in x['article_ids_inview']]), axis=1)
    behaviors['clicked_news'] = behaviors.apply(lambda x: ' '.join([str(i) for i in x['article_id_fixed']]), axis=1)
    behaviors['candidate_news'] = behaviors.apply(lambda x: ' '.join([str(i) for i in x['article_ids_inview']]), axis=1)
    behaviors['impressions'] = behaviors.apply(lambda x: ' '.join([f'{i}-{j}' for i, j in zip(x['article_ids_inview'], x['clicked'].split())]), axis=1)
    behaviors = behaviors[['user_id', 'clicked_news', 'candidate_news', 'clicked', 'impressions']]
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
    behaviors = behaviors.rename(columns={'user_id': 'user'})
    behaviors.to_csv(target_path, sep='\t', index=False, columns=['user', 'clicked_news', 'candidate_news', 'clicked'])

def parse_val_behaviors(behaviors_path, history_path, target_path, articles_path, target_category):
    """
    Parse behaviors file in evaluation set.
    Only keep impressions where candidate news is of the target category.
    Args:
        behaviors_path: source behaviors file
        history_path: source history file
        target_path: target behaviors file
        articles_path: path to articles parquet
        target_category: category to keep
    """
    behaviors = pd.read_parquet(behaviors_path)
    histories = pd.read_parquet(history_path)
    behaviors = behaviors.merge(histories, on='user_id', how='left')
    behaviors = behaviors[['impression_id', 'user_id', 'impression_time', 'article_id_fixed', 
                           'article_ids_inview', 'article_ids_clicked']]
    articles = pd.read_parquet(articles_path)
    # Only keep news IN the target category
    print(articles[articles['category_str'] == target_category].category)
    target_category_news_ids = set(articles[articles['category_str'] == target_category]['article_id'])
    behaviors['article_id_fixed'] = behaviors['article_id_fixed'].apply(
        lambda x: [e for e in x if e in target_category_news_ids])
    behaviors['article_ids_inview'] = behaviors['article_ids_inview'].apply(
        lambda x: [e for e in x if e in target_category_news_ids])
    behaviors['article_ids_clicked'] = behaviors['article_ids_clicked'].apply(
        lambda x: [e for e in x if e in target_category_news_ids])
    behaviors = behaviors[behaviors['article_id_fixed'].apply(len) > 0]
    behaviors = behaviors[behaviors['article_ids_inview'].apply(len) > 0]
    behaviors = behaviors[behaviors['article_ids_clicked'].apply(len) > 0]
    behaviors['clicked'] = behaviors.apply(lambda x: ' '.join(['1' if i in x['article_ids_clicked'] else '0' for i in x['article_ids_inview']]), axis=1)
    behaviors['clicked_news'] = behaviors.apply(lambda x: ' '.join([str(i) for i in x['article_id_fixed']]), axis=1)
    behaviors['candidate_news'] = behaviors.apply(lambda x: ' '.join([str(i) for i in x['article_ids_inview']]), axis=1)
    behaviors['impressions'] = behaviors.apply(lambda x: ' '.join([f'{i}-{j}' for i, j in zip(list(x['article_ids_inview']), list(x['clicked'].split()))]), axis=1)
    behaviors = behaviors.rename(columns={'user_id': 'user', 'impression_time': 'time'})
    behaviors.to_csv(target_path, sep='\t', index=False, columns=['impression_id', 'user', 'time', 'clicked_news', 'impressions', 'candidate_news', 'clicked'], header=False)

def parse_news(news_path, target):
    news = pd.read_parquet(news_path)
    news = news[['article_id', 'category', 'subcategory', 'title', 'body', 'url']]
    # clean title and body columns
    news['title'] = news['title'].str.replace(r'\n', ' ', regex=True)
    news['title'] = news['title'].str.replace(r'\t', ' ', regex=True)
    news['title'] = news['title'].str.replace(r'\s+', ' ', regex=True)
    news['title'] = news['title'].str.strip()
    news['body'] = news['body'].str.replace(r'\n', ' ', regex=True)
    news['body'] = news['body'].str.replace(r'\t', ' ', regex=True)
    news['body'] = news['body'].str.replace(r'\s+', ' ', regex=True)
    news['body'] = news['body'].str.strip()
    news['subcategory'] = news['subcategory'].apply(lambda x: x[0] if len(x) > 0 else 0)
    news.to_csv(target, sep='\t', index=False, header=False)

if __name__ == '__main__':

    target_category = 'sport'  # Define your target category here

    base = f'./data/ebnerd_{target_category}'
    train_dir = f'{base}/train'
    val_dir = f'{base}/val'
    test_dir = f'{base}/test'

    parse_behaviors(path.join(train_dir, 'behaviors.parquet'),
                    path.join(train_dir, f'history.parquet'),
                    path.join(train_dir, f'behaviors_parsed.tsv'),
                    path.join(train_dir, 'articles.parquet'),
                    target_category)
    parse_val_behaviors(path.join(val_dir, 'behaviors.parquet'),
                    path.join(val_dir, f'history.parquet'),
                    path.join(val_dir, f'behaviors_parsed_0.tsv'),
                    path.join(val_dir, 'articles.parquet'),
                    target_category)
    parse_news(path.join(train_dir, 'articles.parquet'),
              path.join(train_dir, f'news.tsv'))
