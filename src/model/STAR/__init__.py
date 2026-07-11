"""
STAR (Sparse Text Approach for Recommendation; Tigunova et al., CIKM 2024).
A sparse bi-encoder that encodes user history as concatenated title text
and candidate news as category+title+abstract text.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from model.layers import DotProductClickPredictor
from model.utils import freeze_bert_layers
from utils import LambdaScheduler


class TextEncoder(nn.Module):
    """
    Shared text encoder for both user history and candidate news.
    Produces SPLADE-style sparse vectors from text input.
    """
    def __init__(self, config, plm, device):
        super().__init__()
        self.config = config
        self.plm = plm
        self.device = device
        self.hidden_size = plm.config.hidden_size
        self.vocab_size = plm.config.vocab_size
        
    def forward(self, input_ids, attention_mask):
        """
        Encode text to sparse vector.
        """
        outputs = self.plm(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True
        )
        
        logits = outputs.logits
        sparse_sequence = torch.log1p(torch.relu(logits))
        
        mask = attention_mask.to(self.device).unsqueeze(-1)
        sparse_sequence = sparse_sequence * mask
        
        sparse_vec = torch.max(sparse_sequence, dim=1)[0]
        return sparse_vec


class STAR(nn.Module):
    """STAR (Tigunova et al., CIKM 2024).

    Sparse bi-encoder for recommendation that represents a user as the
    concatenated text of their clicked news.
    """
    def __init__(self, model_config, dataset_config, device):
        super().__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.device = device
        
        pretrained_name = (
            dataset_config.get('star_pretrained_model_name') or
            model_config.get('pretrained_model_name') or
            dataset_config.get('pretrained_model_name') or
            'naver/splade-cocondenser-ensembledistil'
        )
        self.plm = AutoModelForMaskedLM.from_pretrained(
            pretrained_name,
            output_hidden_states=True,
            trust_remote_code=True
        )
        
        freeze_layers = model_config.get('freeze_bert_layers', 0)
        freeze_bert_layers(self.plm, freeze_layers)
        
        self.text_encoder = TextEncoder(model_config, self.plm, device)
        self.click_predictor = DotProductClickPredictor()
        
        self.target_lambda_n = float(model_config.get('lambda_n', 0.0))
        self.initial_lambda = float(model_config.get('lambda_initial', 0.0))
        self.lambda_scheduler = LambdaScheduler(
            target_lambda_n=self.target_lambda_n,
            initial_lambda_n=self.initial_lambda,
            warmup_steps=int(model_config.get('lambda_scheduler_steps', 50000)),
            enable_warmup=bool(model_config.get('lambda_scheduler_warmup', True))
        )
        
        self.target_lambda_u = float(model_config.get('lambda_u', 0.0))
        self.lambda_u_scheduler = LambdaScheduler(
            target_lambda_n=self.target_lambda_u,
            initial_lambda_n=self.initial_lambda,
            warmup_steps=int(model_config.get('lambda_scheduler_steps', 50000)),
            enable_warmup=bool(model_config.get('lambda_scheduler_warmup', True))
        )

        self.temperature = nn.Parameter(torch.ones(1))
        
    def _calculate_flops_regularizer(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates FLOPS regularizer for sparse vectors.
        """
        vectors = torch.relu(vectors)
        
        if vectors.dim() == 3:
            avg_activation = vectors.mean(dim=(0, 1))
        elif vectors.dim() == 2:
            avg_activation = vectors.mean(dim=0)
        else:
            avg_activation = vectors.view(-1, vectors.shape[-1]).mean(dim=0)
            
        return torch.sum(torch.pow(avg_activation, 2))
    
    def forward(self, user_input, candidate_input, candidate_mask=None):
        batch_size = user_input['input_ids'].shape[0]
        num_candidates = len(candidate_input)
        
        user_vec = self.text_encoder(
            user_input['input_ids'],
            user_input['attention_mask']
        )
        
        cand_input_ids = torch.cat([c['input_ids'] for c in candidate_input], dim=0)
        cand_attention_mask = torch.cat([c['attention_mask'] for c in candidate_input], dim=0)
        
        candidate_vecs_flat = self.text_encoder(cand_input_ids, cand_attention_mask)
        candidate_vecs = candidate_vecs_flat.view(num_candidates, batch_size, -1).transpose(0, 1)
        
        click_prob = self.click_predictor(candidate_vecs, user_vec) / self.temperature
        
        current_lambda_n = self.lambda_scheduler.get_lambda_values()
        current_lambda_u = self.lambda_u_scheduler.get_lambda_values()
        candidate_flops = self._calculate_flops_regularizer(candidate_vecs)
        user_flops = self._calculate_flops_regularizer(user_vec)
        total_reg = current_lambda_n * candidate_flops + current_lambda_u * user_flops
        
        reg_info = {
            'total': total_reg,
            'candidate_flops': candidate_flops,
            'history_news_flops': torch.tensor(0.0),
            'user_flops': user_flops,
            'candidate_weighted': current_lambda_n * candidate_flops,
            'history_news_weighted': torch.tensor(0.0),
            'user_weighted': current_lambda_u * user_flops,
            'current_lambda_n': current_lambda_n,
            'current_lambda_u': current_lambda_u,
            'avg_user_scaling': 1.0,
            'avg_history_scaling': 1.0,
            'avg_history_length': 0.0,
        }
        
        return click_prob, reg_info
    
    def get_news_vector(self, news):
        return self.text_encoder(
            news['input_ids'],
            news['attention_mask']
        )
    
    def get_user_vector(self, user_text_input, mask=None):
        if isinstance(user_text_input, torch.Tensor):
            user_vec = user_text_input
        else:
            user_vec = self.text_encoder(
                user_text_input['input_ids'],
                user_text_input['attention_mask']
            )
            
        topk = getattr(self, 'topk_sparsity_queries', None)
        if topk is not None:
            k = int(topk)
            if 0 < k < user_vec.size(-1):
                _, indices = torch.topk(user_vec, k=k, dim=-1)
                mask_topk = torch.zeros_like(user_vec).to(user_vec.device).scatter_(-1, indices, 1.0)
                user_vec = user_vec * mask_topk
                
        return user_vec
    
    def get_prediction(self, news_vector, user_vector):
        return self.click_predictor(news_vector, user_vector)
