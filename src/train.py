"""
Training script for news recommendation models.
"""
# Standard library imports
import argparse
import gc
from typing import Any, Dict

# Third-party imports
import yaml
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
from model.LSNR_CAT import LSNR_CAT
from datasets import BaseDataset
from ebnerd_datasets import EbnerdBaseDataset
from evaluate import eval_retrieval, build_news_vectors
from utils import EarlyStopping, log_metrics, save_checkpoint, set_seeds, get_device, flatten_dict, get_arg_parser  


def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Set random seeds for reproducibility
set_seeds()
# Parse command-line arguments
args = get_arg_parser().parse_args()
# Load configuration
config = load_config()
# Get device (CPU or CUDA)
device = get_device()

# Model registry for easy model selection
MODEL_REGISTRY = {
    'nrms': NRMSBERT,
    'naml': NAMLBERT,
    'lstur': LSTURBERT,
    'lsnr': LSNR,
    'lsnr_cat': LSNR_CAT
}

# Project name mapping for wandb logging
PROJECT_NAMES = {
    'mind': 'mind',
    'mind_sports': 'mind_sports',
    'mind_news': 'mind_news',
    'ebnerd': 'ebnerd',
    'ebnerd_nyheder': 'ebnerd_nyheder',
    'ebnerd_sport': 'ebnerd_sport'
}

def evaluate_and_log(model: nn.Module, config: dict, early_stopping: EarlyStopping, args: argparse.Namespace, cold_threshold=0) -> tuple[bool, bool, float]:
    """
    Run evaluation, log metrics, and check early stopping.
    Returns (should_stop_training, get_better, val_auc).
    """
    model.eval()
    # Build news and user vectors once for both evaluations
    news2vector, news2category = build_news_vectors(model, config, args, cold_threshold=cold_threshold)

    # Set filter_category based on dataset. This is used for the retrieval evaluation in out-of-distribution settings.
    if args.dataset == 'mind_sports':
        filter_category = 156
    elif args.dataset == 'mind_news':
        filter_category = 254
    elif args.dataset == 'ebnerd_nyheder':
        filter_category = 1
    elif args.dataset == 'ebnerd_sport':
        filter_category = 10
    else:
        filter_category = None
        
    # Run retrieval evaluation
    ret_auc, ret_mrr, ret_ndcg5, ret_ndcg10, ret_ndcg20, ret_ndcg50, ret_ndcg100, ret_hr5, ret_hr10, ret_hr20, ret_hr50, ret_hr100, ret_recall5, ret_recall10, ret_recall20, ret_recall50, ret_recall100 = eval_retrieval(
        model, news2vector, news2category, cold_threshold=cold_threshold, args=args, filter_category=filter_category)
    # Log evaluation metrics
    log_metrics({
        "ret_auc": ret_auc,
        "ret_mrr": ret_mrr,
        "ret_ndcg5": ret_ndcg5,
        "ret_ndcg10": ret_ndcg10,
        "ret_ndcg20": ret_ndcg20,
        "ret_ndcg50": ret_ndcg50,
        "ret_ndcg100": ret_ndcg100,
        "ret_hr5": ret_hr5,
        "ret_hr10": ret_hr10,
        "ret_hr20": ret_hr20,
        "ret_hr50": ret_hr50,
        "ret_hr100": ret_hr100,
        "ret_recall5": ret_recall5,
        "ret_recall10": ret_recall10,
        "ret_recall20": ret_recall20,
        "ret_recall50": ret_recall50,
        "ret_recall100": ret_recall100
    })

    # Early stopping based on negative AUC (since lower is better for early stopping)
    early_stop, get_better = early_stopping(-ret_auc)
    model.train()   
    gc.collect()
    torch.cuda.empty_cache()
    return early_stop, get_better, ret_auc

def train(model: nn.Module, config: dict, args: argparse.Namespace, device: torch.device) -> None:
    """
    Main training loop for the model.
    Handles dataset loading, training, evaluation, logging, and checkpointing.
    """
    if args.dataset not in PROJECT_NAMES:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    project_name = f'{PROJECT_NAMES[args.dataset]}_cold{args.cold_threshold}'

    # Prepare wandb config
    args_dict = vars(args)
    flat_config = flatten_dict(config)
    wandb_config = {**flat_config, **args_dict}

    # Initialize wandb
    wandb.init(project=project_name, name=args.model, config=wandb_config)

    model.to(device)
    model.train()

    # Load dataset based on dataset type
    if args.dataset in ['ebnerd', 'ebnerd_nyheder', 'ebnerd_sport']:
        dataset = EbnerdBaseDataset(
            f"{config[args.dataset]['path']}/train/behaviors_parsed.tsv",    
            f"{config[args.dataset]['path']}/train/news.tsv",
            config,
            args)
    else:
        dataset = BaseDataset(
            f"{config[args.dataset]['path']}/train/behaviors_parsed.tsv",    
            f"{config[args.dataset]['path']}/train/news.tsv",
            config,
            args)
    
    # DataLoader for batching
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    early_stopping = EarlyStopping()
    should_stop_training = False
    training_step = 0

    for epoch in range(args.num_epochs):
        if should_stop_training:
            break
        print(f"Epoch {epoch}")
        for batch in tqdm(dataloader):
            training_step += 1
            # Skip empty batches
            if len(batch["clicked_news"]) == 0:
                continue
            # Forward pass and regularization handling by model type
            if args.model in ['nrms', 'naml']:
                y_pred = model(batch["candidate_news"], 
                               batch["clicked_news"],
                               batch["clicked_news_mask"])   
                regularization = 0         
            elif args.model in ['lsnr', 'lsnr_cat']:
                y_pred, regularization = model(batch["candidate_news"], 
                               batch["clicked_news"],
                               batch["clicked_news_mask"])
                regularization = 0  # TODO: Remove this line if model returns real regularization
            elif args.model == 'lstur':
                y_pred = model(batch["candidate_news"], 
                               batch["clicked_news"],
                               batch["clicked_news_mask"],
                               batch["user"])
                regularization = 0

            # Prepare target labels (all zeros, as only first candidate is positive)
            y = torch.zeros(len(y_pred)).long().to(device) 
            
            # Compute loss (add regularization if any)
            loss = criterion(y_pred, y) + regularization

            # Log regularization and learning rate if regularization is nonzero
            if regularization != 0:
                wandb.log({
                    "regularization": regularization.item(),
                })
      
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # Step the learning rate scheduler
            scheduler.step()
            # Log training metrics
            log_metrics({
                "loss": loss.item(),
                "regularization": regularization.item() if isinstance(regularization, torch.Tensor) else regularization
            })
            # Periodic evaluation and checkpointing
            if training_step % args.eval_every == 0:
                early_stop, get_better, val_auc = evaluate_and_log(model, config, early_stopping, args, cold_threshold=args.cold_threshold)
                if early_stop:
                    should_stop_training = True
                    break
                elif get_better:
                    checkpoint_name = f"{args.model}_{args.dataset}_cold{args.cold_threshold}_lr{args.learning_rate}"
                    save_checkpoint(model, optimizer, checkpoint_name, val_auc)

if __name__ == '__main__':
    # Validate model selection
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {args.model}. Available models: {list(MODEL_REGISTRY.keys())}")
    # Instantiate model
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(config[args.model], config[args.dataset], device)
    # Start training
    train(model, config, args, device)