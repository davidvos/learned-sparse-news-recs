import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import get_config
from utils import (
    set_seeds, get_device, load_model_and_checkpoint,
    EarlyStopping, log_metrics, save_checkpoint, prepare_wandb_config,
)
from data.dataset import BaseDataset, STARDataset
from evaluate import evaluate_model

def train():
    config, args = get_config()
    set_seeds(config['global']['seed'])
    device = get_device(args.device)
    
    model, _, _ = load_model_and_checkpoint(args.model_type, args.dataset, config, device, checkpoint_dir=None)
    
    ds_cfg = config['datasets'][args.dataset]
    mod_cfg = config['models'][args.model_type]
    train_cfg = config['training']
    
    ratio = int(mod_cfg.get('negative_sampling_ratio', config['global'].get('negative_sampling_ratio', 4)))
    
    if args.model_type == 'star':
        config['_model_config'] = mod_cfg
        dataset = STARDataset(f"{ds_cfg['path']}/train/behaviors_parsed.tsv", f"{ds_cfg['path']}/train/news.tsv", config, ratio)
    else:
        dataset = BaseDataset(f"{ds_cfg['path']}/train/behaviors_parsed.tsv", f"{ds_cfg['path']}/train/news.tsv", config, ratio)
    
    dataloader = DataLoader(dataset, batch_size=train_cfg['batch_size'], shuffle=True, 
                            num_workers=train_cfg['num_workers'], drop_last=True, 
                            pin_memory=(device.type != 'mps'), collate_fn=getattr(dataset, "collate_fn", None))

    log_cfg = config['logging']['wandb']
    proj = f"candidate_set_{args.dataset}_cold{train_cfg['cold_threshold']}" if log_cfg['project'] == 'auto' else log_cfg['project']
    
    suffix = ""
    if args.model_type == 'lsnr':
        suffix = f"_ln{mod_cfg.get('lambda_n', 0.0)}_lh{mod_cfg.get('lambda_history', mod_cfg.get('lambda_n', 0.0))}_hls{int(mod_cfg.get('use_history_length_scaling', True))}"
    elif args.model_type == 'star':
        suffix = f"_ln{mod_cfg.get('lambda_n', 0.0)}"

    wandb.init(project=proj, name=f"{config['experiment']['name']}{suffix}", 
               config=prepare_wandb_config(config, args.model_type, args.dataset),
               entity=log_cfg['entity'], tags=log_cfg['tags'], notes=log_cfg['notes'])

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=train_cfg.get('early_stopping', {}).get('patience', 5))
    
    step = 0
    for epoch in range(train_cfg['num_epochs']):
        print(f"Epoch {epoch}")
        for batch in tqdm(dataloader):
            step += 1
            if len(batch.get("clicked_news", batch.get("user_input", {}).get("input_ids", []))) == 0: continue
            
            optimizer.zero_grad()
            
            # Forward pass
            if args.model_type == "star":
                out = model(batch["user_input"], batch["candidate_news"], None)
            elif args.model_type in ['lstur', 'npa']:
                out = model(batch["candidate_news"], batch["clicked_news"], batch["clicked_news_mask"], batch["user"])
            else:
                out = model(batch["candidate_news"], batch["clicked_news"], batch["clicked_news_mask"])

            y_pred, reg = out if isinstance(out, tuple) else (out, None)
            loss = criterion(y_pred, torch.argmax(batch["clicked"], dim=1).to(device))
            
            if reg:
                loss += reg.get('total', 0)
                metrics = {f"regularization/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in reg.items()}
                metrics["loss"] = loss.item()
                log_metrics(metrics)
            else:
                log_metrics({"loss": loss.item()})
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg['gradient_clip_norm'])
            optimizer.step()
            
            if hasattr(model, 'lambda_scheduler'):
                model.lambda_scheduler.step()
            
            # Evaluation
            if step % train_cfg['eval_every'] == 0:
                metrics = evaluate_model(model, config, device, split='val', prefix=f"cold{train_cfg['cold_threshold']}_")
                val_score = metrics.get(train_cfg.get('early_stopping', {}).get('monitor', 'ret_auc'), 0.0)
                
                es_val = val_score if train_cfg.get('early_stopping', {}).get('mode', 'max') == 'min' else -val_score
                stop, better = early_stopping(es_val)
                
                if better:
                    name = f"{config['experiment']['name']}_{args.model_type}_{args.dataset}_cold{train_cfg['cold_threshold']}_lr{train_cfg['learning_rate']}{suffix}"
                    save_checkpoint(model, optimizer, name, val_score, train_cfg.get('early_stopping', {}).get('monitor', 'ret_auc'), train_cfg.get('early_stopping', {}).get('mode', 'max'))
                
                if stop: break
        if stop: break

if __name__ == "__main__":
    train()
