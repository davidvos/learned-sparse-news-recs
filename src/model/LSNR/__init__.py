import torch
import torch.nn as nn
from model.base import BaseModel
from transformers import AutoModelForMaskedLM
from model.LSNR.news_encoder import NewsEncoder
from model.LSNR.user_encoder import UserEncoder
from model.layers import DotProductClickPredictor

class LSNR(BaseModel):
    """
    LSNR model for news recommendation using sparse representations.
    Encodes news and user history with PLM-based encoders and predicts clicks.
    Refactored to support SPLADE-style sparse ranking with dense user modeling.
    """
    def _init_plm(self):
        """
        Initialize the Pre-trained Language Model.
        LSNR uses AutoModelForMaskedLM for SPLADE-style sparse representations.
        """
        return AutoModelForMaskedLM.from_pretrained(
            self.dataset_config['pretrained_model_name'], 
            output_hidden_states=True, 
            trust_remote_code=True
        )

    def __init__(self, model_config, dataset_config, device):
        super().__init__(model_config, dataset_config, device)



        self.news_encoder = NewsEncoder(model_config, dataset_config, self.plm, device)
        self.user_encoder = UserEncoder(
            model_config, 
            dataset_config, 
            self.plm, 
            device,
            tied_projection_layer=self.news_encoder.projection
        )
        
        # Click predictor (dot product) - BM25 saturation applied symmetrically here
        self.click_predictor = DotProductClickPredictor(use_bm25_saturation=True)


        # Regularization Hyperparameters
        self.target_lambda_n = float(model_config.get('lambda_n', 0.0))  # For candidate news
        self.target_lambda_history = float(model_config.get('lambda_history', model_config.get('lambda_n', 0.0)))  # For history news
        self.use_history_length_scaling = bool(model_config.get('use_history_length_scaling', True))
        
        # Max history length for scaling
        self.max_history_length = float(dataset_config.get('num_clicked_news_a_user', 50))
        
        # Initialize scheduler for lambda warmup
        from utils import LambdaScheduler
        self.lambda_scheduler = LambdaScheduler(
            target_lambda_n=self.target_lambda_n,
            warmup_steps=int(model_config.get('lambda_scheduler_steps', 50000)),
            enable_warmup=False # Always disabled for LSNR
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def _calculate_flops_regularizer(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates the FLOPS regularizer for a batch of sparse vectors.
        This encourages a balanced index by penalizing frequently occurring terms.
        """
        # Ensure vectors are non-negative (SPLADE output is log1p(relu), so already >= 0)
        vectors = torch.relu(vectors)
        
        # Calculate the mean activation per vocabulary term
        if vectors.dim() == 3:
            # [batch, num_candidates, vocab] -> Average over batch & candidates
            avg_activation = vectors.mean(dim=(0, 1))
        elif vectors.dim() == 2:
            # [batch, vocab] -> Average over batch
            avg_activation = vectors.mean(dim=0)
        elif vectors.dim() == 1:
            avg_activation = vectors
        else:
            avg_activation = vectors.view(-1, vectors.shape[-1]).mean(dim=0)
        
        # The FLOPS loss is the sum of the squared mean activations
        mean_sq = torch.pow(avg_activation, 2)
        flops_loss = torch.sum(mean_sq)
        
        return flops_loss

    def forward(self, candidate_news, clicked_news, clicked_news_mask):
        """
        Forward pass through the LSNR model.
        """
        num_candidates = len(candidate_news)
        num_clicked = len(clicked_news)
        
        batched_candidate = {
            'input_ids': torch.cat([news['input_ids'] for news in candidate_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in candidate_news], dim=0),
        }
        batch_size = batched_candidate['input_ids'].shape[0] // num_candidates

        sparse_cand_batched, _ = self.news_encoder(batched_candidate)
        sparse_candidate_vecs = sparse_cand_batched.view(num_candidates, batch_size, -1).transpose(0, 1)
        
        batched_clicked = {
            'input_ids': torch.cat([news['input_ids'] for news in clicked_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in clicked_news], dim=0),
        }
        sparse_clicked_batched, dense_clicked_batched = self.news_encoder(batched_clicked, return_sparse=True)
        dense_clicked_vecs = dense_clicked_batched.view(num_clicked, batch_size, -1).transpose(0, 1)
        sparse_clicked_vecs = sparse_clicked_batched.view(num_clicked, batch_size, -1).transpose(0, 1)

        # --- 3. Prepare Mask for History ---
        # Normalize mask shape to [batch, num_clicked]
        if isinstance(clicked_news_mask, torch.Tensor):
            mask_bt = clicked_news_mask
        elif isinstance(clicked_news_mask, list):
            if len(clicked_news_mask) > 0 and isinstance(clicked_news_mask[0], torch.Tensor):
                mask_bt = torch.stack(clicked_news_mask, dim=0).transpose(0, 1)
            else:
                mask_bt = torch.tensor(clicked_news_mask)
        else:
            mask_bt = torch.tensor(clicked_news_mask)

        mask_bt = mask_bt.to(self.device, dtype=dense_clicked_vecs.dtype)
        if mask_bt.dim() == 1:
            mask_bt = mask_bt.unsqueeze(0)
            
        # Apply mask and encode user
        expanded_mask = mask_bt.unsqueeze(-1)
        dense_clicked_vecs = dense_clicked_vecs * expanded_mask
        masked_sparse_clicked = sparse_clicked_vecs * expanded_mask
        user_vec = self.user_encoder(dense_clicked_vecs, masked_sparse_clicked, mask=mask_bt)
        
        current_lambda_n = self.lambda_scheduler.get_lambda_values()
        masked_sparse_history = sparse_clicked_vecs * expanded_mask
        
        if self.use_history_length_scaling:
            history_lengths = mask_bt.sum(dim=1).float()
            scaling_factor = torch.tanh(0.1 * torch.clamp(history_lengths - 1, min=0))
            history_scaling = scaling_factor.view(batch_size, 1, 1)
            weighted_history_vecs = masked_sparse_history * history_scaling
            weighted_candidate_vecs = sparse_candidate_vecs

        else:
            weighted_candidate_vecs = sparse_candidate_vecs
            weighted_history_vecs = masked_sparse_history
        
        # Calculate FLOPS regularization
        candidate_flops_reg = self._calculate_flops_regularizer(weighted_candidate_vecs)
        history_news_flops_reg = self._calculate_flops_regularizer(weighted_history_vecs)

        # Calculate weighted terms - history uses separate lambda
        candidate_weighted = current_lambda_n * candidate_flops_reg
        history_news_weighted = self.target_lambda_history * history_news_flops_reg
        
        # Total regularization (no user regularizer)
        total_reg = candidate_weighted + history_news_weighted
        
        # Compute click prediction (Dot product of Sparse User & Sparse Candidate)
        # Apply learnable temperature scaling
        click_prob = self.click_predictor(sparse_candidate_vecs, user_vec) / self.temperature
        
        reg_info = {
            'total': total_reg,
            'candidate_flops': candidate_flops_reg,
            'history_news_flops': history_news_flops_reg,
            'user_flops': torch.tensor(0.0),  # No longer regularized
            'candidate_weighted': candidate_weighted,
            'history_news_weighted': history_news_weighted,
            'user_weighted': torch.tensor(0.0), # No longer regularized
            'current_lambda_n': current_lambda_n,
            'current_lambda_history': self.target_lambda_history,
            'current_lambda_u': 0.0,
            'avg_history_scaling': history_scaling.mean().item() if self.use_history_length_scaling else 1.0,
            'avg_history_length': history_lengths.mean().item() if self.use_history_length_scaling else mask_bt.sum(dim=1).float().mean().item(),
        }

        return click_prob, reg_info


    def get_news_vector(self, news):
        """
        Returns the SPARSE news vector for ranking, and DENSE for user history.
        Returns: (sparse_vec, dense_vec)
        """
        sparse_vec, dense_vec = self.news_encoder(news)
        return sparse_vec, dense_vec

    def get_user_vector(self, input_data, mask=None):
        """
        Generates user vector. 
        Args:
            input_data: 
                - dict: raw news batch
                - Tensor: pre-computed dense vectors [batch, seq, hidden]
                - tuple (Tensor, Tensor): (dense_vecs, sparse_vecs)
            mask: [batch, seq] (1=Real, 0=Padding)
        """
        if isinstance(input_data, tuple):
            dense_vecs, sparse_vecs = input_data
        elif isinstance(input_data, torch.Tensor):
            dense_vecs = input_data
            sparse_vecs = None
        else:
            # Encode raw input to get both
            sparse_vecs, dense_vecs = self.news_encoder(input_data)
            
        # Handle dimensions for dense: if [hidden_size] -> [1, 1, hidden]
        if dense_vecs.dim() == 1:
            dense_vecs = dense_vecs.unsqueeze(0).unsqueeze(0)
        elif dense_vecs.dim() == 2:
             dense_vecs = dense_vecs.unsqueeze(0) # [1, seq_len, hidden]

        # Handle dimensions for sparse
        if sparse_vecs is not None:
            if sparse_vecs.dim() == 1:
                sparse_vecs = sparse_vecs.unsqueeze(0).unsqueeze(0)
            elif sparse_vecs.dim() == 2:
                sparse_vecs = sparse_vecs.unsqueeze(0)

        # Handle mask dimensions
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
        
        return self.user_encoder(dense_vecs, sparse_vecs, mask=mask)

    def get_prediction(self, news_vector, user_vector):
        """
        Get click prediction from sparse vectors.
        """
        return self.click_predictor(news_vector, user_vector) / self.temperature