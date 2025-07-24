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


set_seeds()
args = get_arg_parser().parse_args()
config = load_config()
device = get_device()
print(device)

# Add model registry
MODEL_REGISTRY = {
    'nrms': NRMSBERT,
    'naml': NAMLBERT,
    'lstur': LSTURBERT,
    'lsnr': LSNR,
    'lsnr_cat': LSNR_CAT
}

# Add project name mapping
PROJECT_NAMES = {
    'mind': 'mind',
    'mind_sports': 'mind_sports',
    'mind_news': 'mind_news',
    'ebnerd': 'ebnerd',
    'ebnerd_nyheder': 'ebnerd_nyheder',
    'ebnerd_sport': 'ebnerd_sport'
}

def evaluate_and_log(model: nn.Module, config: dict, early_stopping: EarlyStopping, args: argparse.Namespace, cold_threshold=0) -> tuple[bool, bool, float]:
    """Run evaluation, log metrics, and check early stopping. Returns (should_stop_training, get_better, val_auc)."""
    model.eval()
    # Build news and user vectors once for both evaluations
    news2vector, news2category = build_news_vectors(model, config, args, cold_threshold=cold_threshold)
    # Retrieval evaluation

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
        
    ret_auc, ret_mrr, ret_ndcg5, ret_ndcg10, ret_ndcg20, ret_ndcg50, ret_ndcg100, ret_hr5, ret_hr10, ret_hr20, ret_hr50, ret_hr100, ret_recall5, ret_recall10, ret_recall20, ret_recall50, ret_recall100 = eval_retrieval(model, news2vector, news2category, cold_threshold=cold_threshold, args=args, filter_category=filter_category)
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

    early_stop, get_better = early_stopping(-ret_auc)
    model.train()   
    gc.collect()
    torch.cuda.empty_cache()
    return early_stop, get_better, ret_auc

def train(model: nn.Module, config: dict, args: argparse.Namespace, device: torch.device) -> None:
    """Train the model."""
    if args.dataset not in PROJECT_NAMES:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    project_name = f'{PROJECT_NAMES[args.dataset]}_cold{args.cold_threshold}'

    args_dict = vars(args)
    flat_config = flatten_dict(config)
    wandb_config = {**flat_config, **args_dict}

    wandb.init(project=project_name, name=args.model, config=wandb_config)

    model.to(device)
    model.train()

    # Load dataset based on dataset type
    if args.dataset == 'ebnerd' or args.dataset == 'ebnerd_nyheder' or args.dataset == 'ebnerd_sport':
        dataset = EbnerdBaseDataset(f"{config[args.dataset]['path']}/train/behaviors_parsed.tsv",    
                            f"{config[args.dataset]['path']}/train/news.tsv",
                            config,
                            args)
    else:
        dataset = BaseDataset(f"{config[args.dataset]['path']}/train/behaviors_parsed.tsv",    
                            f"{config[args.dataset]['path']}/train/news.tsv",
                            config,
                            args)
    
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

            if len(batch["clicked_news"]) == 0:
                continue
            
            if args.model == 'nrms' or args.model == 'naml':
                y_pred = model(batch["candidate_news"], 
                               batch["clicked_news"],
                               batch["clicked_news_mask"])   
                regularization = 0         
            elif args.model == 'lsnr' or args.model == 'lsnr_cat':
                y_pred, regularization = model(batch["candidate_news"], 
                               batch["clicked_news"],
                               batch["clicked_news_mask"])
                regularization = 0
            elif args.model == 'lstur':
                y_pred = model(batch["candidate_news"], 
                               batch["clicked_news"],
                               batch["clicked_news_mask"],
                               batch["user"])
                regularization = 0


            y = torch.zeros(len(y_pred)).long().to(device) 
                
            loss = criterion(y_pred, y) + regularization

            # Log regularization and learning rate
            if regularization != 0:
                wandb.log({
                    "regularization": regularization.item(),
                })
      
            optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Step the scheduler
            scheduler.step()
            
            log_metrics({
                "loss": loss.item(),
                "regularization": regularization.item() if isinstance(regularization, torch.Tensor) else regularization
            })
            
            if training_step % args.eval_every == 0:
                early_stop, get_better, val_auc = evaluate_and_log(model, config, early_stopping, args, cold_threshold=args.cold_threshold)
                if early_stop:
                    should_stop_training = True
                    break
                elif get_better:
                    checkpoint_name = f"{args.model}_{args.dataset}_cold{args.cold_threshold}_lr{args.learning_rate}"
                    save_checkpoint(model, optimizer, checkpoint_name, val_auc)

if __name__ == '__main__':
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {args.model}. Available models: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(config[args.model], config[args.dataset], device)
    train(model, config, args, device)