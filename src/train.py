"""
Training script for news recommendation models.
"""
# Standard library imports
import argparse
import gc
import os
import sys
from typing import Any, Dict

# Set tokenizer parallelism to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from torch.utils.data import default_collate

# Local imports
from model.NRMSBERT import NRMSBERT
from model.NAMLBERT import NAMLBERT
from model.LSTURBERT import LSTURBERT
from model.LSNR import LSNR
from datasets import BaseDataset
from adressa_datasets import AdressaBaseDataset
from evaluate import eval_retrieval, build_news_vectors
from utils import EarlyStopping, log_metrics, save_checkpoint, set_seeds, get_device, flatten_dict, prepare_wandb_config
from config import load_experiment_config

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train news recommendation models")
parser.add_argument('experiment', help='Experiment name (e.g., lsnr_baseline)')
parser.add_argument('dataset', nargs='?', default='mind', help='Dataset to use (default: mind)')
args = parser.parse_args()

experiment_name = args.experiment
dataset_type = args.dataset

# Load configuration
try:
    config = load_experiment_config(experiment_name, dataset_type)
except Exception as e:
    print(f"Configuration error: {e}")
    sys.exit(1)

# Set random seeds for reproducibility
set_seeds(config['global']['seed'])

# Get device based on config preference
device = get_device(config['global']['device'])

# Get model type from experiment config
model_type = config['experiment']['model']

# Get model and dataset configurations
model_config = config['models'][model_type]
dataset_config = config['datasets'][dataset_type]

# Model registry for easy model selection
MODEL_REGISTRY = {
    'nrms': NRMSBERT,
    'naml': NAMLBERT,
    'lstur': LSTURBERT,
    'lsnr': LSNR,
}

# Project name mapping for wandb logging
PROJECT_NAMES = {
    'mind': 'mind',
    'ebnerd': 'ebnerd',
    'adressa': 'adressa',
}

def evaluate_and_log(model: nn.Module, config: dict, early_stopping: EarlyStopping, cold_threshold=0, prefix: str = "", apply_early_stopping: bool = True) -> tuple[bool, bool, float]:
    """
    Run evaluation, log metrics, and check early stopping.
    Returns (should_stop_training, get_better, val_auc).
    """
    model.eval()
    # Build news and user vectors for validation set
    news2vector, news2category = build_news_vectors(
        model,
        config,
        dataset_type,
        model_type,
        config['training']['batch_size'],
        config['training']['num_workers'],
        cold_threshold=cold_threshold,
        split='val',
        device=device,
    )

    # Run retrieval evaluation on validation set
    ret_auc, ret_mrr, ret_ndcg5, ret_ndcg10 = eval_retrieval(
        model,
        news2vector,
        news2category,
        cold_threshold=cold_threshold,
        dataset_type=dataset_type,
        model_type=model_type,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        filter_category=None,
        split='val',
        device=device,
        dataset_path=config['datasets'][dataset_type]['path'],
    )
    # Log evaluation metrics with optional prefix
    log_metrics({
        f"{prefix}ret_auc": ret_auc,
        f"{prefix}ret_mrr": ret_mrr,
        f"{prefix}ret_ndcg5": ret_ndcg5,
        f"{prefix}ret_ndcg10": ret_ndcg10,
    })

    # Early stopping based on negative AUC (since lower is better for early stopping)
    if apply_early_stopping:
        early_stop, get_better = early_stopping(-ret_auc)
    else:
        early_stop, get_better = False, False
    model.train()   
    gc.collect()
    torch.cuda.empty_cache()
    return early_stop, get_better, ret_auc

def train(model: nn.Module, config: dict, device: torch.device, model_type: str, dataset_type: str) -> None:
    """
    Main training loop for the model.
    Handles dataset loading, training, evaluation, logging, and checkpointing.
    """
    if dataset_type not in PROJECT_NAMES:
        raise ValueError(f"Invalid dataset: {dataset_type}")
    
    # Get configuration sections
    training_config = config['training']
    dataset_config = config['datasets'][dataset_type]
    model_config = config['models'][model_type]
    logging_config = config['logging']
    
    # Determine project name
    if logging_config['wandb']['project'] == 'auto':
        project_name = f'candidate_set_{PROJECT_NAMES[dataset_type]}_cold{training_config["cold_threshold"]}'
    else:
        project_name = logging_config['wandb']['project']
    
    # Prepare wandb config with proper type conversion
    wandb_config = prepare_wandb_config(config, model_type, dataset_type)

    # Initialize wandb
    wandb.init(
        project=project_name, 
        name=config['experiment']['name'], 
        config=wandb_config,
        entity=logging_config['wandb']['entity'],
        tags=logging_config['wandb']['tags'],
        notes=logging_config['wandb']['notes']
    )

    model.to(device)
    model.train()

    # Load dataset based on dataset type
    if dataset_type in ['ebnerd', 'adressa']:
        dataset = AdressaBaseDataset(
            f"{dataset_config['path']}/train/behaviors_parsed.tsv",    
            f"{dataset_config['path']}/train/news.tsv",
            config)
    else:
        dataset = BaseDataset(
            f"{dataset_config['path']}/train/behaviors_parsed.tsv",    
            f"{dataset_config['path']}/train/news.tsv",
            config)
    
    # DataLoader for batching
    dataloader = DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=training_config['num_workers'],
        drop_last=True,
        pin_memory=True,
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])
    # Removed learning rate scheduler per request
    early_stopping = EarlyStopping(patience=training_config['early_stopping']['patience'])
    should_stop_training = False
    training_step = 0
    
    for epoch in range(training_config['num_epochs']):
        if should_stop_training:
            break
        print(f"Epoch {epoch}")
        for batch in tqdm(dataloader):
            training_step += 1
            # Skip empty batches
            if len(batch["clicked_news"]) == 0:
                continue
            # Forward pass and regularization handling by model type
            if model_type in ['nrms', 'naml']:
                y_pred = model(batch["candidate_news"], 
                               batch["clicked_news"],
                               batch["clicked_news_mask"])   
                regularization = 0         
            elif model_type == 'lsnr':
                y_pred, reg_info = model(batch["candidate_news"], 
                               batch["clicked_news"],
                               batch["clicked_news_mask"])
                regularization = reg_info['total']
            elif model_type == 'lstur':
                y_pred = model(batch["candidate_news"], 
                               batch["clicked_news"],
                               batch["clicked_news_mask"],
                               batch["user"])
                regularization = 0

            # Prepare target labels (all zeros, as only first candidate is positive)
            y = torch.zeros(len(y_pred)).long().to(device) 
            
            # Compute loss
            loss = criterion(y_pred, y) + regularization
            
            # Log regularization and learning rate if regularization is nonzero
            if regularization != 0:
                if model_type == 'lsnr':
                    # Log detailed regularization components for LSNR
                    wandb.log({
                        "regularization/total": reg_info['total'].item(),
                        "regularization/candidate_flops": reg_info['candidate_flops'].item(),
                        "regularization/user_flops": reg_info['user_flops'].item(),
                        "regularization/candidate_weighted": reg_info['candidate_weighted'].item(),
                        "regularization/user_weighted": reg_info['user_weighted'].item(),
                        "lambda/current_lambda_n": reg_info['current_lambda_n'],
                        "lambda/current_lambda_u": reg_info['current_lambda_u'],
                    })
                else:
                    wandb.log({
                        "regularization": regularization.item(),
                    })
      
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_config['gradient_clip_norm'])
            optimizer.step()
            # Learning rate scheduler removed
            # Step the lambda scheduler for LSNR
            if model_type == 'lsnr':
                model.lambda_scheduler.step()
            # Log training metrics
            log_metrics({
                "loss": loss.item(),
                "regularization": regularization.item() if isinstance(regularization, torch.Tensor) else regularization
            })
            
            # Periodic evaluation and checkpointing
            if training_step % training_config['eval_every'] == 0:
                # Primary eval with configured cold threshold (used for early stopping and checkpoints)
                early_stop, get_better, val_auc = evaluate_and_log(
                    model, config, early_stopping,
                    cold_threshold=training_config['cold_threshold'],
                    prefix="cold0_" if training_config['cold_threshold'] == 0 else f"cold{training_config['cold_threshold']}_",
                    apply_early_stopping=True
                )
                # Secondary eval for cold start 1 users (logging only)
                _es2, _gb2, _auc2 = evaluate_and_log(
                    model, config, early_stopping,
                    cold_threshold=1,
                    prefix="cold1_",
                    apply_early_stopping=False
                )
                if early_stop:
                    should_stop_training = True
                    break
                elif get_better:
                    experiment_name = config['experiment']['name']
                    checkpoint_name = (
                        f"{experiment_name}_{model_type}_{dataset_type}"
                        f"_cold{training_config['cold_threshold']}"
                        f"_lr{training_config['learning_rate']}"
                    )
                    save_checkpoint(model, optimizer, checkpoint_name, val_auc)

if __name__ == '__main__':
    # Validate model selection
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model_type}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    # Instantiate model
    model_class = MODEL_REGISTRY[model_type]
    model = model_class(model_config, dataset_config, device)
    
    # Start training
    train(model, config, device, model_type, dataset_type)
