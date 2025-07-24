import pandas as pd
from tqdm import tqdm
from os import path
import random   
import yaml

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

def parse_behaviors(source_path, target_path):
    """
    Parse behaviors file in training set.
    Args:
        source: source behaviors file
        target: target behaviors file
    """
    behaviors = pd.read_table(
        source_path,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    # Replace inplace fillna to avoid chained assignment warning
    behaviors['clicked_news'] = behaviors['clicked_news'].fillna(' ')

    behaviors.clicked_news = behaviors.clicked_news.apply(
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
        target_path,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])
    
def parse_eval_behaviors(source_path, target_path, cold_threshold, split='val'):
    """
    Parse behaviors file in evaluation set.
    Args:
        source: source behaviors file
        target: target behaviors file
        cold_threshold: threshold for cold start
        subsample_frac: optional float (0,1], fraction of data to subsample (default None, no subsampling)
    """
    behaviors = pd.read_table(
        source_path,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    # Replace inplace fillna to avoid chained assignment warning
    behaviors['clicked_news'] = behaviors['clicked_news'].fillna(' ')

    if cold_threshold != 0:
        behaviors = behaviors[behaviors.clicked_news.apply(lambda x: len(x.split())) <= cold_threshold]
        behaviors = behaviors[behaviors.clicked_news.apply(lambda x: len(x.split())) > 0]
    
    behaviors.impressions = behaviors.impressions.str.split()

    behaviors['candidate_news'] = behaviors.impressions.apply(
        lambda x: ' '.join([e.split('-')[0] for e in x]))
    behaviors['clicked'] = behaviors.impressions.apply(
        lambda x: ' '.join([e.split('-')[1] for e in x]))
    
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

    if split == 'val':
        grouped_behaviors = grouped_behaviors.iloc[:int(len(grouped_behaviors) * 0.5)].reset_index(drop=True)
    elif split == 'test':
        grouped_behaviors = grouped_behaviors.iloc[int(len(grouped_behaviors) * 0.5):].reset_index(drop=True)
    
    grouped_behaviors.to_csv(
        target_path,
        sep='\t',
        index=False,
        header=False,
        columns=['impression_id', 'user', 'time', 'clicked_news', 'impressions', 'candidate_news', 'clicked'])


if __name__ == '__main__':

    base = './data/mind'

    train_dir = f'{base}/train'
    val_dir = f'{base}/val'
    test_dir = f'{base}/test'
    
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