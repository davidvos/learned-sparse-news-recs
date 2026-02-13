import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union
from tqdm import tqdm
import wandb
import os
import gc

from utils import (
    EarlyStopping,
    log_metrics,
    save_checkpoint,
    prepare_wandb_config,
)
from data.dataset import BaseDataset, STARDataset
from model.base import BaseModel

# Project name mapping for wandb logging
PROJECT_NAMES = {
    'mind': 'mind',
    'ebnerd': 'ebnerd',
}

class Trainer:
    """
    Trainer class for training news recommendation models.
    """
    def __init__(
        self, 
        model: BaseModel, 
        config: Dict[str, Any], 
        device: torch.device, 
        model_type: str, 
        dataset_type: str
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model_type = model_type
        self.dataset_type = dataset_type
        
        self.training_config = config['training']
        self.dataset_config = config['datasets'][dataset_type]
        self.model_config = config['models'][model_type]
        self.logging_config = config['logging']
        
        # Initialize components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.training_config['learning_rate'])
        
        early_stopping_config = self.training_config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(patience=early_stopping_config.get('patience', 5))
        self.monitor_metric = early_stopping_config.get('monitor', 'ret_auc')
        self.monitor_mode = early_stopping_config.get('mode', 'max')
        
        self.dataset = self._load_dataset()
        self.dataloader = self._create_dataloader()
        
        self._init_wandb()

    def _init_wandb(self):
        # Determine project name
        if self.logging_config['wandb']['project'] == 'auto':
            project_name = f'candidate_set_{PROJECT_NAMES[self.dataset_type]}_cold{self.training_config["cold_threshold"]}'
        else:
            project_name = self.logging_config['wandb']['project']
        
        # Prepare wandb config
        wandb_config = prepare_wandb_config(self.config, self.model_type, self.dataset_type)

        # Prepare suffix based on regularization for LSNR
        naming_suffix = ""
        if self.model_type == 'lsnr':
            ln = self.model_config.get('lambda_n', 0.0)
            lh = self.model_config.get('lambda_history', ln)
            hls = self.model_config.get('use_history_length_scaling', True)
            naming_suffix = f"_ln{ln}_lh{lh}_hls{int(hls)}"
        elif self.model_type == 'star':
            ln = self.model_config.get('lambda_n', 0.0)
            naming_suffix = f"_ln{ln}"
        
        # Initialize wandb
        wandb.init(
            project=project_name, 
            name=f"{self.config['experiment']['name']}{naming_suffix}", 
            config=wandb_config,
            entity=self.logging_config['wandb']['entity'],
            tags=self.logging_config['wandb']['tags'],
            notes=self.logging_config['wandb']['notes']
        )

    def _load_dataset(self):
        # Determine behaviors path
        behaviors_path = f"{self.dataset_config['path']}/train/behaviors_parsed.tsv"
        news_path = f"{self.dataset_config['path']}/train/news.tsv"
        
        global_ratio = int(self.config['global'].get('negative_sampling_ratio', 4))
        model_ratio = int(self.model_config.get('negative_sampling_ratio', global_ratio))

        if self.model_type == 'star':
            # STAR uses STARDataset which returns concatenated text strings
            # Pass model config via config dict for STAR-specific settings
            self.config['_model_config'] = self.model_config 
            return STARDataset(behaviors_path, news_path, self.config, negative_sampling_ratio=model_ratio)
        else:
            return BaseDataset(behaviors_path, news_path, self.config, negative_sampling_ratio=model_ratio)

    def _create_dataloader(self):
        collate = getattr(self.dataset, "collate_fn", None)
        return DataLoader(
            self.dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=self.training_config['num_workers'],
            drop_last=True,
            pin_memory=True,
            collate_fn=collate,
        )

    def train(self, evaluator=None): # Evaluator passed optionally to avoid circular imports if needed
        self.model.train()
        
        should_stop_training = False
        training_step = 0
        
        for epoch in range(self.training_config['num_epochs']):
            if should_stop_training:
                break
            print(f"Epoch {epoch}")
            
            for batch in tqdm(self.dataloader):
                training_step += 1
                
                # Input validation
                if self.model_type == "star":
                    if len(batch["user_input"]["input_ids"]) == 0: continue
                else:
                    if len(batch["clicked_news"]) == 0: continue
                
                # Check if model returns reg_info (LSNR/STAR) or not
                self.optimizer.zero_grad()
                
                # Prepare inputs based on model type expectations
                # This part is a bit messy due to different model signatures, 
                # but we can try to unify using the new BaseModel interface if possible.
                # Currently BaseModel.forward signature is consistent-ish but LSNR/STAR output tuple.
                
                if self.model_type == "star":
                     output = self.model(
                        batch["user_input"],
                        batch["candidate_news"],
                        None
                    )
                elif self.model_type in ['lstur', 'npa']:
                    output = self.model(
                        batch["candidate_news"], 
                        batch["clicked_news"],
                        batch["clicked_news_mask"],
                        batch["user"] # Extra arg for these
                    )
                else:
                    output = self.model(
                        batch["candidate_news"], 
                        batch["clicked_news"],
                        batch["clicked_news_mask"]
                    )

                # Handle output tuple (prediction, reg_info)
                regularization = 0
                reg_info = {}
                
                if isinstance(output, tuple):
                    y_pred, reg_info = output
                    regularization = reg_info.get('total', 0)
                else:
                    y_pred = output

                # Target labels
                y = torch.argmax(batch["clicked"], dim=1).to(self.device)
                
                # Loss
                loss = self.criterion(y_pred, y) + regularization
                
                # Logging
                metrics_to_log = {"loss": loss.item()}
                
                if regularization != 0:
                     metrics_to_log["regularization"] = regularization.item() if isinstance(regularization, torch.Tensor) else regularization
                     # Spread detailed reg info
                     for k, v in reg_info.items():
                         if isinstance(v, torch.Tensor):
                             metrics_to_log[f"regularization/{k}"] = v.item()
                         else:
                             metrics_to_log[f"regularization/{k}"] = v
                             
                     # Special LSNR/STAR metrics
                     if 'current_lambda_n' in reg_info:
                         metrics_to_log["lambda/current_lambda_n"] = reg_info['current_lambda_n']

                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.training_config['gradient_clip_norm'])
                self.optimizer.step()
                
                # Scheduler step
                if hasattr(self.model, 'lambda_scheduler'):
                    self.model.lambda_scheduler.step()

                log_metrics(metrics_to_log)
                
                # Evaluation
                if training_step % self.training_config['eval_every'] == 0:
                    should_stop_training = self._run_evaluation(evaluator, training_step, naming_suffix=self._get_naming_suffix())
                    if should_stop_training:
                        break
        
        return should_stop_training

    def _get_naming_suffix(self):
        # Re-construct suffix for checkpoint naming
        naming_suffix = ""
        if self.model_type == 'lsnr':
            ln = self.model_config.get('lambda_n', 0.0)
            lh = self.model_config.get('lambda_history', ln)
            hls = self.model_config.get('use_history_length_scaling', True)
            naming_suffix = f"_ln{ln}_lh{lh}_hls{int(hls)}"
        elif self.model_type == 'star':
            ln = self.model_config.get('lambda_n', 0.0)
            naming_suffix = f"_ln{ln}"
        return naming_suffix

    def _run_evaluation(self, evaluator, training_step, naming_suffix=""):
        if evaluator is None:
            print("Warning: No evaluator provided, skipping evaluation.")
            return False

        # Primary eval
        metrics = evaluator.evaluate(
            self.model, 
            cold_threshold=self.training_config['cold_threshold'],
            split='val',
            prefix=f"cold{self.training_config['cold_threshold']}_"
        )
        
        # Monitor
        monitor_value = metrics.get(self.monitor_metric)
        if monitor_value is None:
             print(f"Warning: Monitor metric {self.monitor_metric} not found.")
             monitor_value = 0.0

        es_value = monitor_value if self.monitor_mode == 'min' else -monitor_value
        early_stop, get_better = self.early_stopping(es_value)
        
        if get_better:
            checkpoint_name = (
                f"{self.config['experiment']['name']}_{self.model_type}_{self.dataset_type}"
                f"_cold{self.training_config['cold_threshold']}"
                f"_lr{self.training_config['learning_rate']}"
                f"{naming_suffix}"
            )
            save_checkpoint(
                self.model,
                self.optimizer,
                checkpoint_name,
                monitor_value,
                self.monitor_metric,
                self.monitor_mode,
            )
        
        return early_stop
