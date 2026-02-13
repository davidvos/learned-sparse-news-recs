"""
STAR (Sparse Text-based Attentive Ranker) for News Recommendation.

A sparse bi-encoder that encodes user history as concatenated title text
and candidate news as category+title+abstract text. Produces sparse vectors
for efficient retrieval using SPLADE-style projections.

Key differences from LSNR:
- User representation: Concatenated history titles (text) instead of per-article encoding
- Single shared encoder for both user and item towers (Siamese)
- No separate NewsEncoder/UserEncoder - unified TextEncoder
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
        

        
        # Use MLM head for SPLADE-style projection (already exists in PLM)
        # This reuses the pretrained vocabulary projection
        
    def forward(self, input_ids, attention_mask):
        """
        Encode text to sparse vector.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            
        Returns:
            sparse_vec: [batch, vocab_size]
        """
        # Get hidden states from PLM
        outputs = self.plm(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True
        )
        
        # Get logits from MLM head: [batch, seq_len, vocab_size]
        logits = outputs.logits
        
        # SPLADE activation: log(1 + ReLU(x))
        sparse_sequence = torch.log1p(torch.relu(logits))
        
        # Mask padding tokens before max pooling
        mask = attention_mask.to(self.device).unsqueeze(-1)  # [batch, seq, 1]
        sparse_sequence = sparse_sequence * mask
        
        # Max pooling over sequence: [batch, vocab_size]
        sparse_vec = torch.max(sparse_sequence, dim=1)[0]
        
        # No BM25 saturation (controlled by predictor if needed, but disabled for STAR)
        return sparse_vec


class STAR(nn.Module):
    """
    STAR model for news recommendation using sparse text representations.
    
    Encodes user history as concatenated title text and candidate news as
    category+title+abstract. Uses a shared Siamese encoder for both towers.
    Compatible with existing training pipeline (BCE loss, eval_retrieval).
    """
    
    def __init__(self, model_config, dataset_config, device):
        super().__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.device = device
        
        # Load SPLADE or BERT backbone
        pretrained_name = model_config.get(
            'pretrained_model_name',
            dataset_config.get('pretrained_model_name', 'naver/splade-cocondenser-ensembledistil')
        )
        self.plm = AutoModelForMaskedLM.from_pretrained(
            pretrained_name,
            output_hidden_states=True,
            trust_remote_code=True
        )
        
        # Freeze layers if configured
        freeze_layers = model_config.get('freeze_bert_layers', 0)
        freeze_bert_layers(self.plm, freeze_layers)
        
        # Enable gradient checkpointing for memory efficiency
        if model_config.get('gradient_checkpointing', False):
            self.plm.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for STAR PLM")
        
        # Shared text encoder for both user and item (Siamese)
        self.text_encoder = TextEncoder(model_config, self.plm, device)
        
        # Click predictor (dot product with optional BM25 saturation)
        self.click_predictor = DotProductClickPredictor()
        
        # FLOPS regularization
        self.target_lambda_n = float(model_config.get('lambda_n', 0.0))
        self.initial_lambda = float(model_config.get('lambda_initial', 0.0))
        self.lambda_scheduler = LambdaScheduler(
            target_lambda_n=self.target_lambda_n,
            initial_lambda_n=self.initial_lambda,
            warmup_steps=int(model_config.get('lambda_scheduler_steps', 50000)),
            enable_warmup=bool(model_config.get('lambda_scheduler_warmup', True))
        )
        
        # Learnable temperature for dot product scaling
        self.temperature = nn.Parameter(torch.ones(1))
        
    def _calculate_flops_regularizer(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates FLOPS regularizer for sparse vectors.
        Penalizes frequently occurring terms to encourage balanced sparsity.
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
        """
        Forward pass for training.
        
        Args:
            user_input: dict with 'input_ids', 'attention_mask' for user history text
                        Shape: [batch, seq_len]
            candidate_input: list of dicts, each with 'input_ids', 'attention_mask'
                            for candidate news text. Length = 1 + negative_sampling_ratio
            candidate_mask: unused (for API compatibility with LSNR)
            
        Returns:
            click_prob: [batch, num_candidates] click logits
            reg_info: dict with regularization details
        """
        batch_size = user_input['input_ids'].shape[0]
        num_candidates = len(candidate_input)
        
        # 1. Encode user history text -> sparse vector [batch, vocab]
        user_vec = self.text_encoder(
            user_input['input_ids'],
            user_input['attention_mask']
        )
        
        # 2. Encode all candidates in one batch
        # Stack candidates: [num_candidates * batch, seq_len]
        cand_input_ids = torch.cat([c['input_ids'] for c in candidate_input], dim=0)
        cand_attention_mask = torch.cat([c['attention_mask'] for c in candidate_input], dim=0)
        
        # Encode -> [num_candidates * batch, vocab]
        candidate_vecs_flat = self.text_encoder(cand_input_ids, cand_attention_mask)
        
        # Reshape -> [num_candidates, batch, vocab] -> [batch, num_candidates, vocab]
        candidate_vecs = candidate_vecs_flat.view(num_candidates, batch_size, -1).transpose(0, 1)
        
        # 3. Click prediction via dot product
        # [batch, num_candidates]
        click_prob = self.click_predictor(candidate_vecs, user_vec) / self.temperature
        
        # 4. Regularization
        current_lambda_n = self.lambda_scheduler.get_lambda_values()
        
        # STAR doesn't have a distinct "history news" vector set to regularize separately in the same way LSNR does
        # (it concatenates text). It produces a single user vector.
        # User requested removing lambda_u (user vector regularization).
        # So we ONLY regularize candidate news.
        
        candidate_flops = self._calculate_flops_regularizer(candidate_vecs)
        
        # Total reg is just candidate reg
        total_reg = current_lambda_n * candidate_flops
        
        reg_info = {
            'total': total_reg,
            'candidate_flops': candidate_flops,
            'history_news_flops': torch.tensor(0.0),
            'user_flops': torch.tensor(0.0), # No longer regularized
            'candidate_weighted': total_reg,
            'history_news_weighted': torch.tensor(0.0), 
            'user_weighted': torch.tensor(0.0), # No longer regularized
            'current_lambda_n': current_lambda_n,
            'current_lambda_u': 0.0,
            'avg_user_scaling': 1.0,  
            'avg_history_scaling': 1.0,
            'avg_history_length': 0.0, 
        }
        
        return click_prob, reg_info
    
    def get_news_vector(self, news):
        """
        Get sparse vector for a news article (for evaluation).
        
        Args:
            news: dict with 'input_ids', 'attention_mask' for news text
            
        Returns:
            sparse_vec: [batch, vocab_size] sparse vector
        """
        sparse_vec = self.text_encoder(
            news['input_ids'],
            news['attention_mask']
        )
        return sparse_vec
    
    def get_user_vector(self, user_text_input, mask=None):
        """
        Get sparse vector for user history (for evaluation).
        
        Args:
            user_text_input: dict with 'input_ids', 'attention_mask' 
                            OR pre-computed sparse vectors (for API compatibility)
            mask: unused
            
        Returns:
            sparse_vec: [batch, vocab_size]
        """
        if isinstance(user_text_input, torch.Tensor):
            # Already a sparse vector (passed from evaluation loop)
            return user_text_input
            
        return self.text_encoder(
            user_text_input['input_ids'],
            user_text_input['attention_mask']
        )
    
    def get_prediction(self, news_vector, user_vector):
        """
        Get click prediction from sparse vectors (for evaluation).
        """
        return self.click_predictor(news_vector, user_vector)
