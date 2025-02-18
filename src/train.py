import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
import gc
import numpy as np
import random

from model.NRMSBERT import NRMSBERT
from model.NAMLBERT import NAMLBERT
from model.LSTURBERT import LSTURBERT

from model.LSNR import LSNR
from model.LSNR_CAT import LSNR_CAT

from datasets import BaseDataset
from evaluate import eval
from utils import EarlyStopping


# set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

with open("./config.yaml", "r") as file:
    config = yaml.safe_load(file)

model_name = config[args.model]['model']
pretrained_model_name = config[args.model]['pretrained_model_name']

device = (
    torch.device("mps") if torch.backends.mps.is_built() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
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
    'ebnerd': 'ebnerd'
}

def train(model: nn.Module) -> None:
    if args.dataset not in PROJECT_NAMES:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    project_name = PROJECT_NAMES[args.dataset]
    wandb.init(project=project_name, name=args.model, config=config[args.model])

    model.to(device)
    model.train()

    dataset = BaseDataset(f"{config[args.dataset]['path']}/train/behaviors_parsed.tsv",
                        f"{config[args.dataset]['path']}/train/news.tsv",
                        config,
                        args)
    
    dataloader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=config['num_workers'],
                            drop_last=True,
                            pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=config['learning_rate'])
    

    early_stopping = EarlyStopping()
    should_stop_training = False

    training_step = 0
    
    for epoch in range(config['num_epochs']):
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
                        
            optimizer.step()

            wandb.log({"loss": loss.item()})

            if training_step % 1000 == 0:
                model.eval()
                val_auc, val_mrr, val_ndcg5, val_ndcg10 = eval(model, config, cold_threshold=0)
                wandb.log({"val_auc": val_auc, "val_mrr": val_mrr, "val_ndcg5": val_ndcg5, "val_ndcg10": val_ndcg10})

                cold_val_auc, cold_val_mrr, cold_val_ndcg5, cold_val_ndcg10 = eval(model, config, cold_threshold=2)
                wandb.log({"cold_val_auc": cold_val_auc, "cold_val_mrr": cold_val_mrr, "cold_val_ndcg5": cold_val_ndcg5, "cold_val_ndcg10": cold_val_ndcg10})

                if args.model == 'lsnr' or args.model == 'lsnr_cat':
                    early_stop, get_better = early_stopping(-cold_val_auc)
                else:
                    early_stop, get_better = early_stopping(-val_auc)

                if early_stop:
                    should_stop_training = True
                    break
                elif get_better:

                    checkpoint_name = f"{model_name}_{config[args.model]['model']}_{args.dataset}"

                    if not os.path.exists(f"./checkpoints/{checkpoint_name}/"):
                        os.makedirs(f"./checkpoints/{checkpoint_name}/")
                    for file in os.listdir(f"./checkpoints/{checkpoint_name}/"):
                        os.remove(f"./checkpoints/{checkpoint_name}/{file}")
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': wandb.run.step,
                            'early_stop_value': -val_auc
                        }, f"./checkpoints/{checkpoint_name}/ckpt-{wandb.run.step}.pth")

                model.train()

                gc.collect()
                torch.cuda.empty_cache()
                
if __name__ == '__main__':
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {args.model}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(config[args.model], device)
    train(model)