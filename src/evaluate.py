import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from multiprocessing import Pool
from utils import calculate_single_user_metric, latest_checkpoint
import argparse
import torch.nn.functional as F

from datasets import NewsDataset, BehaviorsDataset, UserDataset

from model.SEQ_LSNR import SEQ_LSNR
from model.NRMSBERT import NRMSBERT

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

tokenizer = AutoTokenizer.from_pretrained(config[args.model]['pretrained_model_name'])

def eval(model, config, cold_threshold):

    with torch.no_grad():

        dataset_path = f"data/{args.dataset}"

        news_dataset = NewsDataset(f"{dataset_path}/val/news.tsv", pretrained_model_name, args.model)
        news_dataloader = DataLoader(news_dataset,
                                    batch_size=config['batch_size'] * 16,
                                    shuffle=False,
                                    num_workers=config['num_workers'],
                                    drop_last=False,
                                    pin_memory=True)
        
        news2vector = {}
        news2text = {}

        for batch in tqdm(news_dataloader):
            news_ids = batch['id']
            news_text = batch['text']

            if args.model == 'naml' or args.model == 'lstur' or args.model == 'seq_lsnr_cat':
                batch = {
                    'input_ids': batch['input_ids'].squeeze(1).to(device),
                    'attention_mask': batch['attention_mask'].squeeze(1).to(device),
                    'category': batch['category'].to(device),
                    'subcategory': batch['subcategory'].to(device)
                }
            
            news_vectors = model.get_news_vector(batch)
            news_vectors = news_vectors.detach().cpu()

            for news_id, news_vector in zip(news_ids, news_vectors):
                if args.dataset == 'mind' or args.dataset == 'mind_demo' or args.dataset == 'mind_demo_warm50' or args.dataset == 'mind_warm50' or args.dataset == 'mind_all' or args.dataset == 'mind_all_warm50':
                    news2vector[str(news_id)] = news_vector
                else:
                    news2vector[str(news_id.item())] = news_vector

            
            for news_id, text in zip(news_ids, news_text):
                news2text[str(news_id)] = text

        if news2vector:
            vector_dim = next(iter(news2vector.values())).shape[0]
            news2vector['PADDED_NEWS'] = torch.zeros(vector_dim)

        news2text['PADDED_NEWS'] = ''
        
        user_dataset = UserDataset(f"{dataset_path}/val/behaviors_parsed_{cold_threshold}.tsv")

        user_dataloader = DataLoader(user_dataset,
                                    batch_size=config['batch_size'] * 16,
                                    shuffle=False,
                                    num_workers=config['num_workers'],
                                    drop_last=False,
                                    pin_memory=True)
        
        user_vectors = {}
        for batch in tqdm(user_dataloader, desc="Processing users"):

            # Vector-based user modeling
            clicked_vectors = torch.stack([
                torch.stack([news2vector.get(str(n), news2vector['PADDED_NEWS']) 
                            for n in news_ids])
                for news_ids in batch['clicked_news']
            ]).transpose(0, 1).to(device)
            
            if args.model in ['lstur', 'lsturlsrec']:
                vecs = model.get_user_vector(batch['user'].to(device), clicked_vectors)
            else:
                vecs = model.get_user_vector(clicked_vectors)
            
            vecs = vecs.cpu()
            for uid, vec in zip(batch['id'], vecs):
                user_vectors[str(uid.item())] = vec

            del clicked_vectors, vecs
            
            torch.cuda.empty_cache()
                    
        # Final evaluation
        behaviors_dataset = BehaviorsDataset(f"{dataset_path}/val/behaviors_parsed_{cold_threshold}.tsv")
        behaviors_loader = DataLoader(behaviors_dataset,
                                      shuffle=False,
                                     batch_size=1,
                                     num_workers=config['num_workers'],
                                     pin_memory=True)
        
        tasks = []
        for batch in tqdm(behaviors_loader, desc="Evaluating"):
            candidates = [n[0].split('-')[0] for n in batch['impressions']]
            user_id = str(batch['impression_id'].item())
            
            cand_vectors = torch.stack([news2vector[c] for c in candidates]).to(device)
            user_vector = user_vectors[user_id].to(device)
            
            scores = model.get_prediction(cand_vectors, user_vector).cpu().numpy()
            labels = [int(n[0].split('-')[1]) for n in batch['impressions']]
            tasks.append((labels, scores))
            
            # Cleanup GPU memory
            del cand_vectors, user_vector
            torch.cuda.empty_cache()

    with Pool(processes=config['num_workers']) as pool:
        results = pool.map(calculate_single_user_metric, tasks)

    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T

    return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(ndcg10s)

if __name__ == '__main__':
    model = NRMSBERT(config[args.model], device)

    checkpoint_path = latest_checkpoint(f'./checkpoints/nrms_nrms_mind_demo_warm50/')
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    cold_threshold = 0
    print(f"Evaluating {args.model} on {args.dataset} with cold_threshold {cold_threshold}")
    

    auc, mrr, ndcg5, ndcg10 = eval(model, config, cold_threshold)
    print(f'cold_threshold: {cold_threshold}')

    print(
        f'AUC: {auc:.4f}\nMRR: {mrr:.4f}\nnDCG@5: {ndcg5:.4f}\nnDCG@10: {ndcg10:.4f}'
    )