import pandas as pd
from tqdm import tqdm
import os
import glob
import json
import random
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict, List, Optional, Set
from config import load_config

def _noop():
    return None

def process_single_file(file_path):
    """
    Process a single JSON file and extract interactions and news metadata.
    Returns tuple of (interactions_list, news_metadata_dict).
    """
    interactions = []
    news_metadata = {}

    # Extract date from filename (YYYYMMDD format)
    filename = os.path.basename(file_path)
    inferred_date = filename if filename.isdigit() and len(filename) == 8 else None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # A valid interaction requires a user, an article ID, and engagement (activeTime > 0)
                    if 'userId' in data and 'id' in data and data.get('activeTime', 0) > 0:
                        article_id = data['id']
                        interactions.append({
                            'user': data['userId'],
                            'article': article_id,
                            'time': data['time'],
                            'date': inferred_date
                        })

                        # Store news metadata if article is new
                        if article_id not in news_metadata:
                            category, subcategory = '', ''
                            if 'category1' in data and data['category1']:
                                cats = data['category1'].split('|')
                                category = cats[0] if len(cats) > 0 else ''
                                subcategory = cats[1] if len(cats) > 1 else ''

                            news_metadata[article_id] = {
                                'title': data.get('title', ''),
                                'abstract': data.get('summary', ''),
                                'category': category,
                                'subcategory': subcategory,
                                'url': data.get('url', ''),
                                'title_entities': '',
                                'abstract_entities': ''
                            }
                except (json.JSONDecodeError, AttributeError):
                    continue  # Skip malformed lines or data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], {}
    
    return interactions, news_metadata

def create_impressions_from_clicks(clicks_df, all_articles, recent_articles_lookup, num_negatives):
    """
    Create impression data from click data by sampling negatives per event.

    Args:
        clicks_df (pd.DataFrame): DataFrame with user clicks (user, article, time)
        all_articles (list): List of all available articles for negative sampling
        recent_articles_lookup (dict): Mapping of date string -> list of recent article ids
        num_negatives (int): Number of negatives to sample per impression

    Returns:
        pd.DataFrame: DataFrame with impression data (user, article, time, clicked)
    """
    impressions = []

    for user, group in clicks_df.groupby('user'):
        # For each click, create an impression with sampled negatives
        for _, row in group.sort_values('time').iterrows():
            clicked_article = str(row['article'])
            time = row['time']
            raw_date = row.get('date') if 'date' in row else None
            event_date = str(raw_date) if raw_date is not None and pd.notna(raw_date) else None

            # Sample negatives.
            negatives = _sample_negatives_for_event(
                event_date,
                num_negatives,
                {clicked_article},
                recent_articles_lookup,
                all_articles,
            )

            # Create impression: clicked article + 19 negatives
            impression_articles = [clicked_article] + negatives
            random.shuffle(impression_articles)

            # Add impression record
            impressions.append({
                'user': user,
                'article': clicked_article,  # The clicked article
                'time': time,
                'impression_articles': impression_articles  # Candidate set for this impression
            })

    return pd.DataFrame(impressions)

def _derive_dates_from_time(time_series: pd.Series) -> pd.Series:
    """
    Robustly derive YYYYMMDD strings from heterogeneous timestamps.
    Supports ISO strings, UNIX seconds, and UNIX milliseconds.
    """
    # First, try generic parsing (handles ISO-like strings)
    dt_generic = pd.to_datetime(time_series, errors='coerce', utc=True)

    # Then try numeric parsing for rows still NaT
    numeric = pd.to_numeric(time_series, errors='coerce')
    dt_result = dt_generic.copy()

    # Heuristic: large values are likely milliseconds
    ms_mask = dt_generic.isna() & numeric.notna() & (numeric > 1e12)
    if ms_mask.any():
        dt_ms = pd.to_datetime(numeric[ms_mask], unit='ms', utc=True, errors='coerce')
        dt_result.loc[ms_mask] = dt_ms

    # Smaller positive values are likely seconds
    s_mask = dt_result.isna() & numeric.notna() & (numeric >= 0)
    if s_mask.any():
        dt_s = pd.to_datetime(numeric[s_mask], unit='s', utc=True, errors='coerce')
        dt_result.loc[s_mask] = dt_s

    return dt_result.dt.strftime('%Y%m%d')


def _build_recent_article_lookup(interactions_df: pd.DataFrame, ordered_dates: List[str], window_days: int) -> Dict[str, List[str]]:
    """Precompute candidate pools of recent articles for each date."""
    window_days = max(1, int(window_days))
    articles_by_date: Dict[str, List[str]] = {}
    for date, group in interactions_df.groupby('date'):
        articles_by_date[str(date)] = group['article'].astype(str).unique().tolist()

    date_to_idx = {date: idx for idx, date in enumerate(ordered_dates)}
    lookup: Dict[str, List[str]] = {}
    for date in ordered_dates:
        idx = date_to_idx[date]
        start_idx = max(0, idx - window_days + 1)
        candidate_dates = ordered_dates[start_idx: idx + 1]
        seen: dict[str, None] = {}
        for cand_date in candidate_dates:
            for article in articles_by_date.get(cand_date, []):
                if article not in seen:
                    seen[article] = None
        lookup[date] = list(seen.keys())
    return lookup


def _sample_negatives_for_event(event_date: Optional[str],
                                k: int,
                                excluded_set: Set[str],
                                recent_lookup: Dict[str, List[str]],
                                all_articles: List[str]) -> List[str]:
    """Sample negatives prioritizing recent articles with fallback to global pool."""
    negatives: List[str] = []
    candidate_pool = []
    if event_date is not None and event_date in recent_lookup:
        candidate_pool = [a for a in recent_lookup[event_date] if a not in excluded_set]
    random.shuffle(candidate_pool)
    for article in candidate_pool:
        if len(negatives) >= k:
            break
        negatives.append(article)

    if len(negatives) < k:
        remaining = [a for a in all_articles if a not in excluded_set and a not in set(negatives)]
        random.shuffle(remaining)
        needed = max(0, k - len(negatives))
        negatives.extend(remaining[:needed])

    return negatives[:k]

def parse_adressa_logs(data_path, max_workers=None, debug_limit=None):
    """
    Parses all raw Adressa JSON log files to extract user interactions and news metadata.
    Uses parallel processing for faster file loading.

    Args:
        data_path (str): Path to the directory containing the JSON files.
        max_workers (int): Number of parallel workers. If None, uses cpu_count().
        debug_limit (int): If set, only process first N files for debugging.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame of user interactions (user, article, time).
            - pd.DataFrame: DataFrame of news metadata (article_id, category, etc.).
    """
    # Find all files in the data directory (files are named by date YYYYMMDD)
    all_files = []
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if os.path.isfile(item_path) and item.isdigit() and len(item) == 8:
            # This is a date file (YYYYMMDD), add it to the list
            all_files.append(item_path)
    if not all_files:
        raise FileNotFoundError(f"No data files found in the specified directory: {data_path}")
    
    # Sort files by date directory for consistent ordering
    all_files.sort()
    
    # Debug limit for faster testing
    if debug_limit is not None:
        all_files = all_files[:debug_limit]
        print(f"DEBUG: Processing only first {len(all_files)} files")
    
    print(f"Found {len(all_files)} files to process.")

    # Process files in parallel
    if max_workers is None:
        max_workers = min(cpu_count(), 8)  # Cap at 8 to avoid memory issues

    max_daily_files = int(os.environ.get('ADRESSA_MAX_DAYS', 0))
    if max_daily_files > 0 and len(all_files) > max_daily_files:
        print(f"Limiting raw log parsing to the first {max_daily_files} day files.")
        all_files = all_files[:max_daily_files]

    print(f"Processing files with {max_workers} parallel workers...")
    with Pool(max_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, all_files),
            total=len(all_files),
            desc="Reading raw log files"
        ))

    # Combine results from all files
    all_interactions = []
    all_news_metadata = {}
    
    for interactions, news_metadata in results:
        all_interactions.extend(interactions)
        all_news_metadata.update(news_metadata)

    print(f"Processed {len(all_interactions)} interactions and {len(all_news_metadata)} unique articles")

    interactions_df = pd.DataFrame(all_interactions)
    if len(interactions_df) == 0:
        raise ValueError("No valid interactions found in the data")
        
    interactions_df.sort_values(by='time', inplace=True)
    interactions_df.drop_duplicates(subset=['user', 'article', 'time'], inplace=True)
    
    # Ensure date exists for every row: prefer directory-derived date, otherwise derive from timestamp
    derived_dates = _derive_dates_from_time(interactions_df['time'])
    if 'date' not in interactions_df.columns:
        interactions_df['date'] = derived_dates
    else:
        interactions_df['date'] = interactions_df['date'].where(interactions_df['date'].notna(), derived_dates)
    # Normalize to string
    interactions_df['date'] = interactions_df['date'].astype(str)

    # Build news dataframe with MIND schema columns
    news_list = []
    for k, v in all_news_metadata.items():
        news_list.append({
            'id': k,
            'category': v.get('category', ''),
            'subcategory': v.get('subcategory', ''),
            'title': v.get('title', ''),
            'abstract': v.get('abstract', ''),
            'url': v.get('url', ''),
            'title_entities': v.get('title_entities', ''),
            'abstract_entities': v.get('abstract_entities', '')
        })
    news_df = pd.DataFrame(news_list, columns=['id','category','subcategory','title','abstract','url','title_entities','abstract_entities'])

    return interactions_df, news_df


def process_train_split(users,
                        history_df,
                        impressions_df,
                        all_articles,
                        config,
                        output_path,
                        recent_articles_lookup,
                        train_negatives: int):
    """
    Generates sequential training samples from the first five days of clicks.
    Histories grow incrementally and the next clicked item is used as the target.

    Args:
        users (list): List of user IDs for this split.
        history_df (pd.DataFrame): DataFrame of historical interactions.
        impressions_df (pd.DataFrame): DataFrame of impressions (click rows from day 6)
        all_articles (list): List of all unique article IDs for negative sampling.
        config (dict): Configuration dictionary.
        output_path (str): Path to save the processed training file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        history_subset = history_df[history_df['user'].isin(users)]
        impressions_subset = impressions_df[impressions_df['user'].isin(users)]

        user_groups_hist = history_subset.groupby('user', sort=False)
        user_groups_impr = impressions_subset.groupby('user', sort=False)

        available_users = sorted(
            set(user_groups_hist.groups.keys()) |
            set(user_groups_impr.groups.keys())
        )

        exclude_history = config['global'].get('negative_sampling_exclude_history', True)
        history_len = int(config['datasets']['adressa']['num_clicked_news_a_user'])

        for user in tqdm(available_users, desc="Processing train users"):
            combined_events = []
            order_idx = 0

            if user in user_groups_hist.groups:
                group_hist = user_groups_hist.get_group(user).sort_values('time')
                for row in group_hist.itertuples():
                    article_id = str(row.article)
                    if not article_id:
                        continue
                    try:
                        ts = float(row.time)
                    except (TypeError, ValueError):
                        continue
                    event_date = getattr(row, 'date', None)
                    if pd.notna(event_date):
                        event_date = str(event_date)
                    else:
                        event_date = None
                    combined_events.append((ts, order_idx, article_id, event_date))
                    order_idx += 1

            if user in user_groups_impr.groups:
                group_impr = user_groups_impr.get_group(user).sort_values('time')
                for row in group_impr.itertuples():
                    article_id = str(row.article)
                    if not article_id:
                        continue
                    try:
                        ts = float(row.time)
                    except (TypeError, ValueError):
                        continue
                    event_date = getattr(row, 'date', None)
                    if pd.notna(event_date):
                        event_date = str(event_date)
                    else:
                        event_date = None
                    combined_events.append((ts, order_idx, article_id, event_date))
                    order_idx += 1

            if len(combined_events) < 2:
                continue

            combined_events.sort(key=lambda x: (x[0], x[1]))

            rolling_hist = []
            for idx, (_, _, article_id, event_date) in enumerate(combined_events):
                if idx == 0:
                    rolling_hist.append(article_id)
                    continue

                history_slice = rolling_hist[-history_len:]
                if not history_slice:
                    rolling_hist.append(article_id)
                    continue

                history_str = ' '.join(history_slice)
                positive_article = article_id

                excluded = {positive_article}
                if exclude_history:
                    excluded.update(history_slice)

                negatives = _sample_negatives_for_event(
                    event_date,
                    train_negatives,
                    excluded,
                    recent_articles_lookup,
                    all_articles,
                )

                candidates = [positive_article] + [str(n) for n in negatives]
                labels = ['1'] + ['0'] * len(negatives)

                candidate_str = ' '.join(candidates)
                clicked_str = ' '.join(labels)
                f.write(f"{user}\t{history_str}\t{candidate_str}\t{clicked_str}\n")

                rolling_hist.append(article_id)


def process_eval_split(users, history_df, impressions_df, all_articles, config, output_path, split_name, cold_threshold=0):
    """
    Generates evaluation data using impressions with 20 articles.
    Each impression contains 1 clicked article + 19 negatives.

    Args:
        users (list): List of user IDs for this split.
        history_df (pd.DataFrame): DataFrame of historical interactions.
        impressions_df (pd.DataFrame): DataFrame of impressions with impression_articles column.
        all_articles (list): List of all unique article IDs for negative sampling.
        config (dict): Configuration dictionary.
        output_path (str): Path to save the processed evaluation file.
        split_name (str): Name of the split ('val' or 'test').
    """
    # Write MIND-style parsed eval behaviors with extra columns candidate_news and clicked
    with open(output_path, 'w', encoding='utf-8') as f:
        user_groups_hist = history_df[history_df['user'].isin(users)].groupby('user')
        user_groups_impr = impressions_df[impressions_df['user'].isin(users)].groupby('user')
        impression_id_counter = 1

        for user, group_impr in tqdm(user_groups_impr, desc=f"Processing {split_name} users"):
            group_impr = group_impr.sort_values('time')
            hist_list = []
            if user in user_groups_hist.groups:
                group_hist = user_groups_hist.get_group(user).sort_values('time')
                hist_list = group_hist['article'].astype(str).tolist()
            history_len = int(config['datasets']['adressa']['num_clicked_news_a_user'])
            min_history_len = int(config['datasets']['adressa'].get('min_history_for_eval', 1))
            if len(hist_list) > history_len:
                hist_list = hist_list[-history_len:]

            # Cold filtering: include only users with history length <= cold_threshold (and > 0) if cold_threshold > 0
            if cold_threshold != 0:
                if not (0 < len(hist_list) <= cold_threshold):
                    continue

            # Skip impressions with no history to avoid empty-history evaluation rows
            if len(hist_list) < max(1, min_history_len):
                continue

            history_str = ' '.join(hist_list)

            for _, row in group_impr.iterrows():
                positive_article = str(row['article'])
                impression_articles = row['impression_articles']  # List of 20 articles
                
                # Create impression items: article-label format
                impression_items = []
                for article in impression_articles:
                    label = '1' if str(article) == positive_article else '0'
                    impression_items.append(f"{article}-{label}")
                
                # Shuffle impression items
                random.shuffle(impression_items)
                impressions_str = ' '.join(impression_items)
                candidate_news = ' '.join([e.split('-')[0] for e in impression_items])
                clicked = ' '.join([e.split('-')[1] for e in impression_items])

                time = row['time']

                f.write(f"{impression_id_counter}\t{user}\t{time}\t{history_str}\t{impressions_str}\t{candidate_news}\t{clicked}\n")
                impression_id_counter += 1

def create_mappings(output_path, users, categories):
    """
    Creates and saves user2int.tsv and category2int.tsv files.

    Args:
        output_path (str): The base directory to save the mapping files.
        users (list): A list of all unique user IDs.
        categories (list): A list of all unique category strings.
    """
    print("Creating mapping files...")
    # Create user2int.tsv
    user2int = {user: i + 1 for i, user in enumerate(users)}
    user2int_df = pd.DataFrame(user2int.items(), columns=['user', 'int'])
    user2int_df.to_csv(os.path.join(output_path, 'user2int.tsv'), sep='\t', index=False, header=False)

    # Create category2int.tsv
    category2int = {cat: i + 1 for i, cat in enumerate(categories)}
    category2int_df = pd.DataFrame(category2int.items(), columns=['category', 'int'])
    category2int_df.to_csv(os.path.join(output_path, 'category2int.tsv'), sep='\t', index=False, header=False)


def main():
    """
    Main function to orchestrate the preprocessing pipeline.
    """
    # 1. Load Configuration (experiment YAML for consistency with training)
    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(os.environ.get('PREPROCESS_CONFIG', root / 'config.yaml'))
    config = load_config(str(cfg_path))
    
    global_config = config['global']
    dataset_config = config['datasets']['adressa']

    # 2. Create Output Directories (aligned with training dataset path)
    base_out = config['datasets']['adressa']['path']
    train_dir = os.path.join(base_out, 'train')
    val_dir = os.path.join(base_out, 'val')
    test_dir = os.path.join(base_out, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 3. Parse Raw Data (with parallel processing)
    # Seed RNGs for reproducibility of negative sampling/impressions
    seed = int(global_config.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)

    # Check for debug mode via environment variable
    debug_limit = int(os.environ.get('ADRESSA_DEBUG_LIMIT', 0)) or None
    raw_path = os.environ.get('ADRESSA_RAW_PATH', global_config.get('data_path'))
    if raw_path is None:
        raise ValueError("Set ADRESSA_RAW_PATH or global.data_path to raw Adressa logs")
    interactions_df, news_df = parse_adressa_logs(raw_path, max_workers=4, debug_limit=debug_limit)
    
    # 4. Build splits using configurable windows similar to MIND
    unique_dates = [d for d in interactions_df['date'].dropna().unique().tolist() if d is not None]
    unique_dates = sorted(unique_dates)
    print(f"Parsed unique dates: {len(unique_dates)} found. First few: {unique_dates[:10]}")
    if len(unique_dates) == 0:
        raise ValueError("No valid dates found in Adressa data.")

    history_days = max(0, int(dataset_config.get('history_days', 21)))
    train_days = max(1, int(dataset_config.get('train_days', 7)))
    val_days = max(0, int(dataset_config.get('val_days', 1)))
    test_days = max(0, int(dataset_config.get('test_days', 1)))
    total_required_days = history_days + train_days + val_days + test_days
    if len(unique_dates) < total_required_days:
        raise ValueError(
            f"Not enough days in raw data. Required {total_required_days}, but found {len(unique_dates)}."
        )

    selected_dates = unique_dates[:total_required_days]
    history_dates = selected_dates[:history_days]
    train_dates = selected_dates[history_days: history_days + train_days]
    val_dates = selected_dates[history_days + train_days: history_days + train_days + val_days]
    test_dates = selected_dates[history_days + train_days + val_days: total_required_days]

    print(
        "Configured windows — history: {} days, train: {} days, val: {} days, test: {} days ({} total).".format(
            len(history_dates), len(train_dates), len(val_dates), len(test_dates), len(selected_dates)
        )
    )

    selected_set = set(selected_dates)
    interactions_selected = interactions_df[interactions_df['date'].isin(selected_set)].copy()
    interactions_selected['user'] = interactions_selected['user'].astype(str)
    interactions_selected['article'] = interactions_selected['article'].astype(str)

    if interactions_selected.empty:
        raise ValueError("No interactions remain after selecting configured date ranges.")

    used_article_raw = sorted(interactions_selected['article'].unique())
    news_df = news_df[news_df['id'].astype(str).isin(used_article_raw)].copy()

    # Map users/articles to incremental Uxxx/Nxxx
    all_users_raw = sorted(interactions_selected['user'].unique())
    user2mapped = {u: f"U{idx + 1}" for idx, u in enumerate(all_users_raw)}
    news2mapped = {n: f"N{idx + 1}" for idx, n in enumerate(used_article_raw)}

    interactions_selected['user'] = interactions_selected['user'].map(user2mapped)
    interactions_selected['article'] = interactions_selected['article'].map(news2mapped)
    news_df['id'] = news_df['id'].astype(str).map(news2mapped)

    # Drop articles that lost mapping
    news_df = news_df[news_df['id'].notna()].copy()

    # Filter users by minimum interactions after mapping
    min_interactions = int(global_config.get('min_interactions', 0))
    if min_interactions > 0:
        user_counts = interactions_selected['user'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index.tolist()
        interactions_selected = interactions_selected[interactions_selected['user'].isin(valid_users)].copy()
    else:
        valid_users = interactions_selected['user'].unique().tolist()
    valid_users = sorted(valid_users)

    if interactions_selected.empty:
        raise ValueError("No interactions remain after filtering by minimum interactions.")

    used_articles_mapped = sorted(interactions_selected['article'].unique())
    news_df = news_df[news_df['id'].isin(used_articles_mapped)].copy()

    # Save news with shared article set across splits
    news_df.to_csv(os.path.join(train_dir, 'news.tsv'), sep='\t', index=False, header=False)
    news_df.to_csv(os.path.join(val_dir, 'news.tsv'), sep='\t', index=False, header=False)
    news_df.to_csv(os.path.join(test_dir, 'news.tsv'), sep='\t', index=False, header=False)
    print(f"Saved news.tsv with {len(news_df)} unique articles across selected dates.")

    # Helper to subset interactions by date bucket
    def subset_by_dates(dates: List[str]) -> pd.DataFrame:
        if not dates:
            return pd.DataFrame(columns=interactions_selected.columns)
        return interactions_selected[interactions_selected['date'].isin(dates)].copy()

    history_df = subset_by_dates(history_dates)
    train_df = subset_by_dates(train_dates)
    val_df = subset_by_dates(val_dates)
    test_df = subset_by_dates(test_dates)

    # Derived histories for evaluation splits
    if not history_df.empty or not train_df.empty:
        val_history_df = pd.concat([history_df, train_df], ignore_index=True)
    else:
        val_history_df = pd.DataFrame(columns=interactions_selected.columns)

    if not val_history_df.empty or not val_df.empty:
        test_history_df = pd.concat([val_history_df, val_df], ignore_index=True)
    else:
        test_history_df = val_history_df.copy()

    all_articles = sorted(interactions_selected['article'].unique().tolist())
    neg_window_days = int(dataset_config.get('neg_window_days', 3))
    recent_articles_lookup = _build_recent_article_lookup(interactions_selected, selected_dates, neg_window_days)
    train_negatives = int(dataset_config.get('train_negative_sampling_ratio', config['global'].get('negative_sampling_ratio', 4)))
    eval_candidate_size = int(dataset_config.get('eval_candidate_size', 20))
    eval_negatives = max(1, eval_candidate_size - 1)

    print(
        f"Negative sampling — train negatives: {train_negatives}, eval candidate size: {eval_candidate_size}"
    )

    # 5. Create training split (behaviors_parsed.tsv)
    train_users = sorted(set(history_df['user'].tolist()) | set(train_df['user'].tolist()))
    if train_users:
        process_train_split(
            train_users,
            history_df,
            train_df,
            all_articles,
            config,
            os.path.join(train_dir, 'behaviors_parsed.tsv'),
            recent_articles_lookup,
            train_negatives,
        )
    else:
        open(os.path.join(train_dir, 'behaviors_parsed.tsv'), 'w', encoding='utf-8').close()

    # 6. Create evaluation splits with configurable negatives
    cold_thresholds = [0, 1, 2, 3, 4, 5]
    cfg_cold = config['training'].get('cold_threshold', 1)
    if cfg_cold not in cold_thresholds:
        cold_thresholds.append(cfg_cold)

    # Validation split
    if not val_df.empty:
        print("Building validation impressions...")
        val_impressions_df = create_impressions_from_clicks(
            val_df,
            all_articles,
            recent_articles_lookup,
            eval_negatives,
        )
        val_users = sorted(val_impressions_df['user'].unique().tolist())
        for ct in sorted(cold_thresholds):
            process_eval_split(
                val_users,
                val_history_df,
                val_impressions_df,
                all_articles,
                config,
                os.path.join(val_dir, f'behaviors_parsed_cold{ct}.tsv'),
                'val',
                cold_threshold=ct,
            )
    else:
        for ct in sorted(cold_thresholds):
            open(os.path.join(val_dir, f'behaviors_parsed_cold{ct}.tsv'), 'w', encoding='utf-8').close()

    # Test split
    if not test_df.empty:
        print("Building test impressions...")
        test_impressions_df = create_impressions_from_clicks(
            test_df,
            all_articles,
            recent_articles_lookup,
            eval_negatives,
        )
        test_users = sorted(test_impressions_df['user'].unique().tolist())
        for ct in sorted(cold_thresholds):
            process_eval_split(
                test_users,
                test_history_df,
                test_impressions_df,
                all_articles,
                config,
                os.path.join(test_dir, f'behaviors_parsed_cold{ct}.tsv'),
                'test',
                cold_threshold=ct,
            )
    else:
        for ct in sorted(cold_thresholds):
            open(os.path.join(test_dir, f'behaviors_parsed_cold{ct}.tsv'), 'w', encoding='utf-8').close()

    # 7. Create Mapping Files
    all_categories = pd.concat([news_df['category'], news_df['subcategory']]).dropna().astype(str).unique().tolist()
    create_mappings(train_dir, valid_users, all_categories)
    create_mappings(val_dir, valid_users, all_categories)
    create_mappings(test_dir, valid_users, all_categories)

    print("Preprocessing complete.")


if __name__ == '__main__':
    main()
