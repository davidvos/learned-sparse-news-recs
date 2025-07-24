import pandas as pd
import random
import yaml
from os import path
from tqdm import tqdm

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

def filter_out_category(news_df, category):
    return news_df[news_df['category'] != category]

def filter_in_category(news_df, category):
    return news_df[news_df['category'] == category]

def parse_train_behaviors(source_path, target_path, news_path, target_category):
    """
    Parse and filter training behaviors: remove all news of the target category from histories and candidates.
    Remove samples with no history or candidates left.
    """
    news = pd.read_table(
        news_path,
        names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
        usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
    )
    # Only keep news not in the target category
    allowed_news_ids = set(news[news['category'] != target_category]['id'])

    behaviors = pd.read_table(
        source_path,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    # Replace inplace fillna to avoid chained assignment warning
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

    behaviors = behaviors.explode('impressions').dropna(subset=["impressions"]).reset_index(drop=True)
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (' '.join([e.split('-')[0] for e in x]), ' '.join([e.split('-')[1] for e in x]))).tolist())

    behaviors.to_csv(
        target_path,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])

def parse_eval_behaviors(source_path, val_target_path, test_target_path, news_path, target_category):
    """
    Parse and filter evaluation behaviors: only keep impressions where candidate news is of the target category.
    Remove samples with no history or candidates left.
    Then group by user, aggregate impressions, and split into validation and test sets (first half for val, second half for test).
    """
    news = pd.read_table(
        news_path,
        names=['id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
        usecols=['id', 'category', 'subcategory', 'title', 'abstract'],
    )
    # Only keep news in the target category
    target_news_ids = set(news[news['category'] == target_category]['id'])

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

    behaviors.impressions = behaviors.impressions.str.split()

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
    target_category = 'news'  # Only define target category

    base = f'./data/mind_{target_category}'
    train_dir = f'{base}/train'
    val_dir = f'{base}/val'
    test_dir = f'{base}/test'

    parse_train_behaviors(
        path.join(train_dir, 'behaviors.tsv'),
        path.join(train_dir, 'behaviors_parsed.tsv'),
        path.join(train_dir, 'news.tsv'),
        target_category
    )

    parse_eval_behaviors(
        path.join(val_dir, 'behaviors.tsv'),
        path.join(val_dir, 'behaviors_parsed_0.tsv'),
        path.join(test_dir, 'behaviors_parsed_0.tsv'),
        path.join(train_dir, 'news.tsv'),
        target_category
    )