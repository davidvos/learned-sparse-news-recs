"""Shared utilities for training and evaluation.

Includes:

- **Metrics** — AUC, MRR, nDCG@k, HR@k, Recall@k per-user computation.
- **LambdaScheduler** — quadratic warmup schedule for FLOPS regularisation.
- **EarlyStopping** — patience-based training halting.
- **Checkpointing** — save/load with automatic latest-checkpoint discovery.
- **Device selection** — auto-detect CUDA / MPS / CPU.
- **Model registry** — centralised mapping from model name to class.
"""
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score


class LambdaScheduler:
    """
    Scheduler for regularization lambda parameters that implements quadratic warmup.
    Lambda values start at 0 and quadratically increase to target values over specified steps.
    """
    def __init__(
        self, 
        target_lambda_n: float, 
        warmup_steps: int, 
        enable_warmup: bool = True, 
        initial_lambda_n: float = 0.0, 
    ):
        """
        Args:
            target_lambda_n (float): Target lambda value for news regularization
            warmup_steps (int): Number of steps to reach target lambda values
            enable_warmup (bool): Whether to enable warmup (if False, uses target values immediately)
            initial_lambda_n (float): Initial lambda value for news
        """
        self.target_lambda_n = target_lambda_n
        self.initial_lambda_n = initial_lambda_n
        self.warmup_steps = warmup_steps
        self.enable_warmup = enable_warmup
        self.current_step = 0
        
    def step(self) -> None:
        self.current_step += 1
        
    def get_lambda_values(self) -> float:
        """
        Get current lambda values based on the schedule.
        Returns:
            float: current_lambda_n
        """
        if not self.enable_warmup or self.current_step >= self.warmup_steps:
            # After warmup or if disabled, use target values
            return self.target_lambda_n
        else:
            # Quadratic warmup: lambda = initial + (target - initial) * (step / warmup_steps)^2
            progress = self.current_step / self.warmup_steps
            scale_factor = progress ** 2
            current_lambda_n = self.initial_lambda_n + (self.target_lambda_n - self.initial_lambda_n) * scale_factor
            return current_lambda_n


class EarlyStopping:
    """
    Early stopping helper to stop training when validation loss stops improving.
    """
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss: float) -> Tuple[bool, bool]:
        """
        Args:
            val_loss: Validation loss (or negative metric if higher is better).
            
        Returns:
            Tuple[bool, bool]: (early_stop, is_better)
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


def dcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Compute Discounted Cumulative Gain at k."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return float(np.sum(gains / discounts))


def ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0


def mrr_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank."""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return float(np.sum(rr_score) / np.sum(y_true)) if np.sum(y_true) > 0 else 0.0


def hr_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Compute Hit Rate at k."""
    order = np.argsort(y_score)[::-1][:k]
    y_true_topk = np.take(y_true, order)
    return 1.0 if np.sum(y_true_topk) > 0 else 0.0


def recall_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Compute Recall at k."""
    order = np.argsort(y_score)[::-1][:k]
    y_true_topk = np.take(y_true, order)
    relevant = np.sum(y_true)
    if relevant == 0:
        return float('nan')
    return float(np.sum(y_true_topk) / relevant)


def calculate_single_user_metric(pair: Tuple[np.ndarray, np.ndarray]) -> List[float]:
    """
    Calculate metrics for a single user's prediction.
    
    Args:
        pair: Tuple of (y_true, y_score) arrays.
        
    Returns:
        List of metrics in fixed order:
        [AUC, MRR, NDCG@5, NDCG@10, NDCG@20, NDCG@50, NDCG@100,
         HR@5, HR@10, HR@20, HR@50, HR@100,
         Recall@5, Recall@10, Recall@20, Recall@50, Recall@100]
    """
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
        return [float('nan')] * 17


def latest_checkpoint(directory: str) -> Optional[str]:
    """Find the latest checkpoint file in a directory based on step number."""
    if not os.path.exists(directory):
        return None
    
    try:
        all_checkpoints = {
            int(x.split('.')[-2].split('-')[-1]): x
            for x in os.listdir(directory)
            if x.endswith('.pth') and 'ckpt-' in x
        }
    except (ValueError, IndexError):
        # Fallback if naming convention doesn't match
        return None
        
    if not all_checkpoints:
        return None
        
    return os.path.join(directory, all_checkpoints[max(all_checkpoints.keys())])


def log_metrics(metrics: Dict[str, Any]) -> None:
    wandb.log(metrics)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_name: str,
    monitor_value: float,
    monitor_metric: str,
    monitor_mode: str,
) -> None:
    """Save model/optimizer state along with early stopping metadata."""
    checkpoint_dir = f"./checkpoints/{checkpoint_name}/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Clean up old checkpoints to save space (keep only latest)
    for file in os.listdir(checkpoint_dir):
        os.remove(os.path.join(checkpoint_dir, file))
        
    early_stop_value = monitor_value if monitor_mode == 'min' else -monitor_value
    
    # Save new checkpoint
    try:
        step = wandb.run.step if wandb.run else 0
    except Exception:
        step = 0
        
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'monitor_metric': monitor_metric,
            'monitor_mode': monitor_mode,
            'monitor_value': monitor_value,
            'early_stop_value': early_stop_value,
        },
        os.path.join(checkpoint_dir, f"ckpt-{step}.pth")
    )


def set_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Select the best available device based on preference.
    
    Args:
        device_preference: Device preference ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        Selected torch device
    """
    if device_preference == "auto":
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


def prepare_wandb_config(config: Dict[str, Any], model_type: str, dataset_type: str) -> Dict[str, Any]:
    """
    Prepare configuration for WandB logging with proper type conversion and structure.
    
    Args:
        config: Full configuration dictionary
        model_type: Type of model being used
        dataset_type: Type of dataset being used
        
    Returns:
        Properly formatted configuration for WandB
    """
    wandb_config: Dict[str, Any] = {}
    model_config = config.get('models', {}).get(model_type, {})
    
    # Global settings
    wandb_config['global'] = {
        'seed': config['global']['seed'],
        'device': config['global']['device'],
        'negative_sampling_ratio': model_config.get('negative_sampling_ratio', config['global'].get('negative_sampling_ratio', 4))
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


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Recursively flattens a nested dictionary, joining keys with sep."""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



from model.baselines import NRMSBERT, NAMLBERT, LSTURBERT, NPABERT
from model.LSNR import LSNR
from model.STAR import STAR

MODEL_REGISTRY = {
    'nrms': NRMSBERT,
    'naml': NAMLBERT,
    'lstur': LSTURBERT,
    'lsnr': LSNR,
    'star': STAR,
    'npa': NPABERT,
}

def load_model_and_checkpoint(
    model_type: str,
    dataset_type: str,
    config: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: Optional[str] = None
) -> Tuple[torch.nn.Module, Dict[str, Any], int]:
    """
    Load a model and optionally a checkpoint.
    
    Args:
        model_type: Type of model to load (e.g., 'nrms', 'lsnr')
        dataset_type: Type of dataset (e.g., 'mind')
        config: Full configuration dictionary
        device: Device to load model on
        checkpoint_dir: Directory containing checkpoints (optional)
        
    Returns:
        (model, checkpoint_data, start_step)
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    model_class = MODEL_REGISTRY[model_type]
    
    # Extract configs
    model_config = config['models'][model_type]
    dataset_config = config['datasets'][dataset_type]
    
    # Initialize model
    model = model_class(model_config, dataset_config, device).to(device)
    
    checkpoint_data = {}
    start_step = 0
    
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        # Try to find latest checkpoint
        ckpt_path = latest_checkpoint(checkpoint_dir)
        if ckpt_path:
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            # Remap news_encoder.plm.* → plm.* to handle checkpoints saved before
            # NewsEncoder's PLM was merged with the top-level BaseModel PLM.
            # We prefer news_encoder.plm.* (the actually fine-tuned weights) over
            # any top-level plm.* that may have been untouched during training.
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith('news_encoder.plm.'):
                    remapped[k[len('news_encoder.'):]] = v  # news_encoder.plm.x → plm.x
                else:
                    remapped[k] = v
            # news_encoder.plm is the same object as plm; populate both paths so
            # load_state_dict doesn't report news_encoder.plm.* as missing keys.
            for k in list(remapped.keys()):
                if k.startswith('plm.'):
                    remapped['news_encoder.' + k] = remapped[k]
            missing, unexpected = model.load_state_dict(remapped, strict=False)
            if missing:
                print(f"  Checkpoint missing keys (will use init): {missing}")
            if unexpected:
                print(f"  Checkpoint unexpected keys (ignored): {unexpected}")
            checkpoint_data = checkpoint
            start_step = checkpoint.get('step', 0)
        else:
            print(f"No checkpoint found in {checkpoint_dir}, initializing from scratch.")
            
    return model, checkpoint_data, start_step

