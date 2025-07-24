import numpy as np
from sklearn.metrics import roc_auc_score
import os
import torch
import wandb
import argparse
import random

class EarlyStopping:
    def __init__(self, patience=40):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss):
        """
        if you use other metrics where a higher value is better, e.g. accuracy,
        call this with its corresponding negative value
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def hr_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1][:k]
    y_true_topk = np.take(y_true, order)
    return 1.0 if np.sum(y_true_topk) > 0 else 0.0

def recall_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1][:k]
    y_true_topk = np.take(y_true, order)
    relevant = np.sum(y_true)
    if relevant == 0:
        return np.nan
    return np.sum(y_true_topk) / relevant

def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        ndcg20 = ndcg_score(*pair, 20)
        ndcg50 = ndcg_score(*pair, 50)
        ndcg100 = ndcg_score(*pair, 100)
        hr5 = hr_score(*pair, 5)
        hr10 = hr_score(*pair, 10)
        hr20 = hr_score(*pair, 20)
        hr50 = hr_score(*pair, 50)
        hr100 = hr_score(*pair, 100)
        recall5 = recall_score(*pair, 5)
        recall10 = recall_score(*pair, 10)
        recall20 = recall_score(*pair, 20)
        recall50 = recall_score(*pair, 50)
        recall100 = recall_score(*pair, 100)
        return [auc, mrr, ndcg5, ndcg10, ndcg20, ndcg50, ndcg100,
                hr5, hr10, hr20, hr50, hr100,
                recall5, recall10, recall20, recall50, recall100]
    except ValueError:
        return [np.nan] * 17
    

def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])

def log_metrics(metrics: dict) -> None:
    """Log metrics to wandb."""
    wandb.log(metrics)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_name: str, val_auc: float) -> None:
    """Save model and optimizer state to checkpoint."""
    checkpoint_dir = f"./checkpoints/{checkpoint_name}/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    for file in os.listdir(checkpoint_dir):
        os.remove(os.path.join(checkpoint_dir, file))
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': wandb.run.step,
            'early_stop_value': -val_auc
        }, os.path.join(checkpoint_dir, f"ckpt-{wandb.run.step}.pth")
    )

def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--eval_type', type=str, default='ctr', choices=['ctr', 'retrieval', 'both'], help='Evaluation type: ctr, retrieval, or both')
    parser.add_argument('--cold_threshold', type=int, default=1, help='Threshold for cold start evaluation/training')
    return parser

def get_device() -> 'torch.device':
    """Select the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def flatten_dict(d, parent_key='', sep='.'):  # Helper to flatten nested dicts
    """Recursively flattens a nested dictionary, joining keys with sep."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)