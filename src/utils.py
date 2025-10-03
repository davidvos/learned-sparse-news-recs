import os
import random

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score


class LambdaScheduler:
    """
    Scheduler for regularization lambda parameters that implements quadratic warmup.
    Lambda values start at 0 and quadratically increase to target values over specified steps.
    """
    def __init__(self, target_lambda_n, target_lambda_u, warmup_steps, enable_warmup=True):
        """
        Args:
            target_lambda_n (float): Target lambda value for news regularization
            target_lambda_u (float): Target lambda value for user regularization  
            warmup_steps (int): Number of steps to reach target lambda values
            enable_warmup (bool): Whether to enable warmup (if False, uses target values immediately)
        """
        self.target_lambda_n = target_lambda_n
        self.target_lambda_u = target_lambda_u
        self.warmup_steps = warmup_steps
        self.enable_warmup = enable_warmup
        self.current_step = 0
        
    def step(self):
        """Increment the current step"""
        self.current_step += 1
        
    def get_lambda_values(self):
        """
        Get current lambda values based on the schedule.
        Returns:
            tuple: (current_lambda_n, current_lambda_u)
        """
        if not self.enable_warmup or self.current_step >= self.warmup_steps:
            # After warmup or if disabled, use target values
            return self.target_lambda_n, self.target_lambda_u
        else:
            # Quadratic warmup: lambda = target * (step / warmup_steps)^2
            progress = self.current_step / self.warmup_steps
            scale_factor = progress ** 2
            current_lambda_n = self.target_lambda_n * scale_factor
            current_lambda_u = self.target_lambda_u * scale_factor
            return current_lambda_n, current_lambda_u

class EarlyStopping:
    def __init__(self, patience=5):
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

# Legacy argument parser removed - use argparse directly in scripts

def get_device(device_preference: str = "auto") -> 'torch.device':
    """
    Select the best available device based on preference.
    
    Args:
        device_preference: Device preference ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        Selected torch device
    """
    if device_preference == "auto":
        # User preference: CUDA if available, else MPS, else CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise RuntimeError("CUDA requested but not available")
    elif device_preference == "mps":
        if torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            raise RuntimeError("MPS requested but not available")
    elif device_preference == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Invalid device preference: {device_preference}")

def prepare_wandb_config(config: dict, model_type: str, dataset_type: str) -> dict:
    """
    Prepare configuration for WandB logging with proper type conversion and structure.
    
    Args:
        config: Full configuration dictionary
        model_type: Type of model being used
        dataset_type: Type of dataset being used
        
    Returns:
        Properly formatted configuration for WandB
    """
    # Create a clean config structure for WandB
    wandb_config = {}
    
    # Global settings
    wandb_config['global'] = {
        'seed': config['global']['seed'],
        'device': config['global']['device'],
        'negative_sampling_ratio': config['global']['negative_sampling_ratio']
    }
    
    # Training settings
    wandb_config['training'] = {
        'learning_rate': float(config['training']['learning_rate']),
        'batch_size': int(config['training']['batch_size']),
        'num_epochs': int(config['training']['num_epochs']),
        'num_workers': int(config['training']['num_workers']),
        'eval_every': int(config['training']['eval_every']),
        'cold_threshold': int(config['training']['cold_threshold']),
        'eval_split': config['training']['eval_split'],
        'gradient_clip_norm': float(config['training']['gradient_clip_norm']),
        'scheduler': config['training']['scheduler'],
        'early_stopping': config['training']['early_stopping']
    }
    
    # Model settings (only for the current model)
    model_config = config['models'][model_type].copy()
    # Convert string scientific notation to floats for lambda parameters
    if 'lambda_n' in model_config:
        model_config['lambda_n'] = float(model_config['lambda_n'])
    if 'lambda_u' in model_config:
        model_config['lambda_u'] = float(model_config['lambda_u'])
    if 'lambda_scheduler_steps' in model_config:
        model_config['lambda_scheduler_steps'] = int(model_config['lambda_scheduler_steps'])
    
    wandb_config['model'] = model_config
    
    # Dataset settings (only for the current dataset)
    wandb_config['dataset'] = config['datasets'][dataset_type]
    
    # Experiment metadata
    wandb_config['experiment'] = {
        'name': config['experiment']['name'],
        'model': config['experiment']['model'],
        'description': config['experiment']['description'],
        'dataset': dataset_type
    }
    
    return wandb_config


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
