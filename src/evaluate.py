"""Evaluation utilities and CLI for news recommendation models."""

import argparse
import os
import sys

import torch

# Set tokenizer parallelism to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from config import load_experiment_config
from utils import calculate_single_user_metric, get_device, latest_checkpoint, set_seeds

from datasets import NewsDataset, BehaviorsDataset
from adressa_datasets import AdressaNewsDataset, AdressaBehaviorsDataset

from model.LSNR import LSNR
from model.NAMLBERT import NAMLBERT
from model.LSTURBERT import LSTURBERT
from model.NRMSBERT import NRMSBERT

# Model registry for easy model selection
MODEL_REGISTRY = {
    'nrms': NRMSBERT,
    'naml': NAMLBERT,
    'lstur': LSTURBERT,
    'lsnr': LSNR,
}


def load_model_and_checkpoint(
    model_type: str,
    experiment_name: str,
    dataset: str,
    learning_rate: str | None = None,
    checkpoint_dir: str | None = None,
):
    """Load a configured model along with its latest checkpoint."""
    config = load_experiment_config(experiment_name, dataset)

    set_seeds(config['global']['seed'])
    device = get_device(config['global']['device'])

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model_type}. Available models: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(config['models'][model_type], config['datasets'][dataset], device)

    if checkpoint_dir is not None:
        resolved_ckpt_dir = checkpoint_dir
    else:
        lr = learning_rate if learning_rate is not None else config['training']['learning_rate']
        cold_tr = config['training']['cold_threshold']
        checkpoint_name = f"{experiment_name}_{model_type}_{dataset}_cold{cold_tr}_lr{lr}"
        resolved_ckpt_dir = f"./checkpoints/{checkpoint_name}/"

    print(f"🔍 Looking for checkpoints in: {resolved_ckpt_dir}", flush=True)
    checkpoint_path = latest_checkpoint(resolved_ckpt_dir)
    if checkpoint_path is None:
        print(f"❌ No checkpoint file found in {resolved_ckpt_dir}!", flush=True)
        print("💡 Available checkpoint directories:", flush=True)
        checkpoints_base = "./checkpoints/"
        if os.path.exists(checkpoints_base):
            for item in os.listdir(checkpoints_base):
                if os.path.isdir(os.path.join(checkpoints_base, item)):
                    print(f"   📁 {item}", flush=True)
        raise FileNotFoundError(
            f"No checkpoint found for experiment {experiment_name} with dataset {dataset}"
        )

    print(f"✅ Loading checkpoint: {checkpoint_path}", flush=True)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except Exception as exc:
        if "weights_only" in str(exc) or "WeightsUnpickler" in str(exc):
            print("⚠️  Retrying checkpoint load with weights_only=False (trusted source)", flush=True)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            raise

    model.load_state_dict(checkpoint['model_state_dict'])
    print(
        "📊 Checkpoint info - Step: {step}, Epoch: {epoch}".format(
            step=checkpoint.get('step', 'unknown'),
            epoch=checkpoint.get('epoch', 'unknown'),
        ),
        flush=True,
    )

    model.to(device)
    model.eval()
    return model, device, config

def run_evaluation(
    model,
    device,
    config,
    dataset: str,
    eval_split: str,
    cold_thresholds: list[int],
    batch_size: int | None = None,
):
    """
    Entry point used by the CLI to run evaluation.

    Builds news vectors once for the requested split, then evaluates retrieval
    metrics for each requested cold threshold.
    """
    dataset_type = dataset
    model_type = str(config['experiment']['model'])
    effective_batch_size = batch_size if batch_size is not None else int(config['training']['batch_size'])
    num_workers = int(config['training'].get('num_workers', 4))

    # Build news vectors for the split once
    news2vector, news2category = build_news_vectors(
        model=model,
        config=config,
        dataset_type=dataset_type,
        model_type=model_type,
        batch_size=effective_batch_size,
        num_workers=num_workers,
        cold_threshold=0,
        split=eval_split,
        device=device,
    )

    # Compute average sparsity for news vectors (exclude padding)
    news_nonzero = 0
    news_total = 0
    for nid, vec in news2vector.items():
        if nid == 'PADDED_NEWS':
            continue
        v = vec
        news_nonzero += (v != 0).sum().item()
        news_total += v.numel()
    avg_news_sparsity = float(1.0 - (news_nonzero / news_total)) if news_total > 0 else float('nan')

    # Evaluate for each requested cold threshold
    for ct in cold_thresholds:
        ret_auc, ret_mrr, ret_ndcg5, ret_ndcg10, avg_user_sparsity = eval_retrieval(
            model=model,
            news2vector=news2vector,
            news2category=news2category,
            cold_threshold=ct,
            dataset_type=dataset_type,
            model_type=model_type,
            batch_size=effective_batch_size,
            num_workers=num_workers,
            filter_category=None,
            split=eval_split,
            device=device,
            dataset_path=config['datasets'][dataset_type]['path'],
        )
        print({
            "split": eval_split,
            "cold_threshold": ct,
            "ret_auc": ret_auc,
            "ret_mrr": ret_mrr,
            "ret_ndcg5": ret_ndcg5,
            "ret_ndcg10": ret_ndcg10,
            "sparsity/news_avg": avg_news_sparsity,
            "sparsity/user_avg": avg_user_sparsity,
        }, flush=True)

def eval_retrieval(
    model,
    news2vector,
    news2category,
    cold_threshold=0,
    dataset_type=None,
    model_type=None,
    batch_size=16,
    num_workers=4,
    filter_category=None,
    split='val',
    device=None,
    dataset_path=None,
):
    """
    Retrieval evaluation: For each user, rank against all candidate news from the eval/test set 
    and evaluate where the true clicked news ranks.
    Returns a tuple of averaged metrics.
    """
    if dataset_path is None:
        if dataset_type is None:
            raise ValueError("dataset_type must be provided when dataset_path is None")
        dataset_path = f"data/{dataset_type}"

    behaviors_file = os.path.join(dataset_path, split, f"behaviors_parsed_cold{cold_threshold}.tsv")

    if dataset_type == 'adressa':
        behaviors_dataset = AdressaBehaviorsDataset(
            behaviors_file,
            cold_threshold=cold_threshold,
            num_clicked_news_a_user=30,
        )
    else:
        behaviors_dataset = BehaviorsDataset(
            behaviors_file,
            cold_threshold=cold_threshold,
            num_clicked_news_a_user=30,
        )

    behaviors_loader = DataLoader(
        behaviors_dataset,
        shuffle=False,
        batch_size=batch_size * 4,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=behaviors_collate_fn,
    )

    # Stream metrics computation to avoid holding all users' data in memory
    total_users_processed = 0
    num_metric_samples = 0
    sum_auc = 0.0
    sum_mrr = 0.0
    sum_ndcg5 = 0.0
    sum_ndcg10 = 0.0

    # Track average sparsity of user vectors
    user_nonzero = 0
    user_total = 0

    with torch.no_grad():
        for batch in tqdm(behaviors_loader, desc="Retrieval Evaluating"):
            batch_size = len(batch['user'])
            total_users_processed += batch_size

            # Build user history vectors for each user in the batch
            user_history_vectors = torch.stack([
                torch.stack([news2vector[nid] for nid in user_clicked_news])
                for user_clicked_news in batch['clicked_news']
            ]).to(device)

            # Get user vectors from the model
            if model_type == 'lstur':
                user_vectors = model.get_user_vector(batch['user'], user_history_vectors)
            else:
                user_vectors = model.get_user_vector(user_history_vectors)

            # Accumulate sparsity stats for user vectors
            user_nonzero += (user_vectors != 0).sum().item()
            user_total += user_vectors.numel()

            # For each user in the batch, evaluate only on that user's candidate set
            for i, user_id in enumerate(batch['user']):
                # Build this user's candidate list (optionally filter by category)
                user_candidate_pairs = batch['impressions'][i]
                user_candidate_ids = []
                user_labels = []
                for pair in user_candidate_pairs:
                    news_id, label = pair.split('-')
                    if news_id == "PADDED_NEWS":
                        continue
                    if news_id not in news2vector:
                        continue
                    if filter_category is not None and news2category.get(news_id, 0) != filter_category:
                        continue
                    user_candidate_ids.append(news_id)
                    user_labels.append(int(label))
                if len(user_candidate_ids) == 0:
                    continue  # nothing to evaluate for this user

                # Require at least one positive and one negative to compute AUC/MRR/NDCG
                num_pos = sum(user_labels)
                if num_pos == 0 or num_pos == len(user_labels):
                    continue

                # Prepare candidate vectors for this user
                user_candidate_vectors = torch.stack([news2vector[nid] for nid in user_candidate_ids]).to(user_history_vectors.device)
                # Single-user prediction: add batch dimension
                user_vec = user_vectors[i].unsqueeze(0)
                user_candidate_vectors_batched = user_candidate_vectors.unsqueeze(0)
                user_scores = model.get_prediction(user_candidate_vectors_batched, user_vec).squeeze(0)

                labels_np = np.array(user_labels, dtype=np.float32)
                scores_np = user_scores.detach().cpu().numpy()
                # Calculate metrics immediately to avoid storing per-user data
                metrics = calculate_single_user_metric((labels_np, scores_np))
                auc, mrr, ndcg5, ndcg10 = metrics[0], metrics[1], metrics[2], metrics[3]
                # Skip NaN results
                if not (np.isnan(auc) or np.isnan(mrr) or np.isnan(ndcg5) or np.isnan(ndcg10)):
                    sum_auc += float(auc)
                    sum_mrr += float(mrr)
                    sum_ndcg5 += float(ndcg5)
                    sum_ndcg10 += float(ndcg10)
                    num_metric_samples += 1

            # Free up memory
            del user_history_vectors, user_vectors
            torch.cuda.empty_cache()
    

    # Compute averaged metrics and average user sparsity
    avg_user_sparsity = float(1.0 - (user_nonzero / user_total)) if user_total > 0 else float('nan')

    if num_metric_samples == 0:
        return (float('nan'), float('nan'), float('nan'), float('nan'), avg_user_sparsity)

    return (
        sum_auc / num_metric_samples,
        sum_mrr / num_metric_samples,
        sum_ndcg5 / num_metric_samples,
        sum_ndcg10 / num_metric_samples,
        avg_user_sparsity,
    )


def build_news_vectors(model, config, dataset_type, model_type, batch_size, num_workers, cold_threshold=0, split='val', device=None):
    """
    Build news2vector and news2category dictionaries for all news articles in the specified split.
    Handles batching and device placement for efficiency.
    """
    dataset_path = config['datasets'][dataset_type]['path']
    pretrained_model_name = config['datasets'][dataset_type]['pretrained_model_name']
    num_words_title = config['datasets'][dataset_type]['num_words_title']
    num_words_abstract = config['datasets'][dataset_type]['num_words_abstract']
    # Use Adressa-specific dataset when applicable
    if dataset_type == 'adressa':
        news_dataset = AdressaNewsDataset(f"{dataset_path}/{split}/news.tsv", pretrained_model_name, model_type, dataset_type, num_words_title, num_words_abstract)
    else:
        news_dataset = NewsDataset(f"{dataset_path}/{split}/news.tsv", pretrained_model_name, model_type, dataset_type, num_words_title, num_words_abstract)

    news_dataloader = DataLoader(
        news_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )
    news2vector = {}
    news2category = {}

    # Process news in batches
    for batch in tqdm(news_dataloader, desc="Processing news"):
        news_ids = batch['id']
        news_text = batch['text']
        # Prepare batch data for model
        if model_type in ['naml', 'lstur']:
            batch_data = {
                'input_ids': batch['input_ids'].squeeze(1).to(device),
                'attention_mask': batch['attention_mask'].squeeze(1).to(device),
                'category': batch['category'].to(device),
                'subcategory': batch['subcategory'].to(device)
            }
        else:
            batch_data = batch
        # Get news vectors from the model
        news_vectors = model.get_news_vector(batch_data)
        news_vectors = news_vectors.detach().cpu()
        # Store vectors and categories
        for news_id, news_vector, category in zip(news_ids, news_vectors, batch['category']):
            # Support both tensor ids (e.g., mind) and string ids (e.g., adressa)
            try:
                nid = str(news_id.item())
            except Exception:
                nid = str(news_id)
            news2vector[nid] = news_vector
            news2category[nid] = int(category.item())
        del news_vectors, batch_data
        torch.cuda.empty_cache()
        
    # Add a zero vector for padding
    if news2vector:
        vector_dim = next(iter(news2vector.values())).shape[0]
        news2vector['PADDED_NEWS'] = torch.zeros(vector_dim)
        news2category['PADDED_NEWS'] = 0
    return news2vector, news2category


def behaviors_collate_fn(batch):
    """
    Collate function for behaviors DataLoader. Pads clicked_news and clicked_news_mask to the max length in the batch.
    """
    batch_out = {}
    # Find max length for clicked_news in this batch
    max_clicked_len = max(len(item['clicked_news']) for item in batch)
    # Left pad clicked_news and clicked_news_mask to max_clicked_len
    batch_out['clicked_news'] = [
        ['PADDED_NEWS'] * (max_clicked_len - len(item['clicked_news'])) + item['clicked_news']
        for item in batch
    ]
    batch_out['clicked_news_mask'] = torch.tensor(
        [[0] * (max_clicked_len - len(item['clicked_news_mask'])) + item['clicked_news_mask'] for item in batch]
    )
    batch_out['clicked_news_length'] = torch.tensor([item['clicked_news_length'] for item in batch])
    batch_out['user'] = torch.tensor([item['user'] for item in batch], dtype=torch.long)
    batch_out['impressions'] = [item['impressions'] for item in batch]
    return batch_out


def _print_available_checkpoints() -> None:
    print("📁 Available checkpoints:")
    checkpoints_base = "./checkpoints/"
    if not os.path.exists(checkpoints_base):
        print("   ❌ No checkpoints directory found")
        return

    for item in sorted(os.listdir(checkpoints_base)):
        item_path = os.path.join(checkpoints_base, item)
        if not os.path.isdir(item_path):
            continue
        checkpoint_files = [f for f in os.listdir(item_path) if f.endswith('.pth')]
        count = len(checkpoint_files)
        plural = '' if count == 1 else 's'
        print(f"   📂 {item} ({count} checkpoint{plural})")
        for ckpt in sorted(checkpoint_files):
            print(f"      🔹 {ckpt}")


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Evaluate a trained news recommendation model - just specify the model class!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/evaluate.py lsnr                    # Evaluate LSNR on MIND dataset
  python src/evaluate.py nrms --dataset adressa  # Evaluate NRMS on Adressa dataset
  python src/evaluate.py naml --split val        # Evaluate NAML on validation set
  python src/evaluate.py lstur --cold-thresholds 0 1 2  # Evaluate LSTUR on multiple cold thresholds
  python src/evaluate.py lsnr --batch-size 32    # Evaluate LSNR with custom batch size
  python src/evaluate.py --list-checkpoints      # List all available checkpoints
        """
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()

    parser.add_argument(
        'model',
        nargs='?',
        choices=['lsnr', 'nrms', 'naml', 'lstur'],
        help='Model class to evaluate (lsnr, nrms, naml, lstur)'
    )
    parser.add_argument(
        '--dataset',
        default='mind',
        choices=['mind', 'adressa'],
        help='Dataset to evaluate on (default: mind)'
    )
    parser.add_argument(
        '--split',
        default='test',
        choices=['val', 'test'],
        help='Split to evaluate on (default: test)'
    )
    parser.add_argument(
        '--cold-thresholds',
        nargs='*',
        type=int,
        default=None,
        help='Cold thresholds to evaluate (default: [0, 1, 2, 3, 4, 5])'
    )
    parser.add_argument(
        '--lr',
        type=str,
        default=None,
        help='Learning rate to select checkpoint (e.g., 0.0001 or 1e-05)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        default=None,
        help='Explicit checkpoint directory path'
    )
    parser.add_argument(
        '--list-checkpoints',
        action='store_true',
        help='List all available checkpoints and exit'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for evaluation (default: use experiment config)'
    )
    parser.add_argument(
        '--regularization',
        type=str,
        default=None,
        help='Regularization to select checkpoint (e.g., baseline, low, high, very_high)'
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_checkpoints:
        _print_available_checkpoints()
        return 0

    if args.model is None:
        raise SystemExit("Model argument is required unless --list-checkpoints is used.")

    if args.cold_thresholds is None:
        args.cold_thresholds = [0, 1, 2, 3, 4, 5]

    regularization = args.regularization or 'baseline'
    experiment_name = f"{args.model}_{regularization}"

    print(f"🚀 Evaluating {args.model.upper()} model on {args.dataset.upper()} dataset")
    print(f"📁 Using experiment: {experiment_name}")
    print(f"📊 Split: {args.split}")
    if args.cold_thresholds:
        print(f"❄️  Cold thresholds: {args.cold_thresholds}")
    if args.checkpoint_dir:
        print(f"💾 Checkpoint dir: {args.checkpoint_dir}")
    print()

    try:
        model, device, config = load_model_and_checkpoint(
            model_type=args.model,
            experiment_name=experiment_name,
            dataset=args.dataset,
            learning_rate=args.lr,
            checkpoint_dir=args.checkpoint_dir,
        )

        run_evaluation(
            model=model,
            device=device,
            config=config,
            dataset=args.dataset,
            eval_split=args.split,
            cold_thresholds=args.cold_thresholds,
            batch_size=args.batch_size,
        )
        print("✅ Evaluation completed successfully!")
        return 0
    except Exception as exc:  # noqa: BLE001 - surface error to user
        print(f"❌ Evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
