import torch
from transformers import AutoModelForMaskedLM
from model.LSNR.news_encoder import NewsEncoder
from model.LSNR.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor
import torch.nn as nn

class LSNR(nn.Module):
    """
    LSNR model for news recommendation using sparse representations.
    Encodes news and user history with PLM-based encoders and predicts clicks.
    """
    def __init__(self, model_config, dataset_config, device):
        """
        Args:
            model_config (dict): Model-specific configuration.
            dataset_config (dict): Dataset-specific configuration.
            device (torch.device): Device to run the model on.
        """
        super().__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config
        # Load pre-trained language model with hidden states
        self.plm = AutoModelForMaskedLM.from_pretrained(
            dataset_config['pretrained_model_name'], output_hidden_states=True, trust_remote_code=True)
        self.device = device
        # News and user encoders
        self.news_encoder = NewsEncoder(model_config, dataset_config, self.plm, device)
        self.user_encoder = UserEncoder(model_config, dataset_config, self.plm, device)
        # Click predictor (dot product)
        self.click_predictor = DotProductClickPredictor()

        # Add lambda hyperparameters for regularization
        self.target_lambda_n = float(model_config.get('lambda_n', 0.0))  # Target regularization for news vectors
        self.target_lambda_u = float(model_config.get('lambda_u', 0.0))  # Target regularization for user vectors
        
        # Initialize scheduler for lambda warmup
        from utils import LambdaScheduler
        self.lambda_scheduler = LambdaScheduler(
            target_lambda_n=self.target_lambda_n,
            target_lambda_u=self.target_lambda_u,
            warmup_steps=int(model_config.get('lambda_scheduler_steps', 50000)),
            enable_warmup=bool(model_config.get('lambda_scheduler_warmup', True))
        )

    def _calculate_flops_regularizer(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Calculates the FLOPS regularizer for a batch of sparse vectors.
        This encourages a balanced index by penalizing frequently occurring terms.
        """
        # Ensure vectors are non-negative, as weights should be positive
        vectors = torch.relu(vectors)
        
        # Calculate the mean activation per vocabulary term
        # Handle input shapes:
        #  - [batch, vocab]
        #  - [batch, num_candidates, vocab]
        if vectors.dim() == 3:
            # Average over batch and candidate dimensions -> [vocab]
            avg_activation = vectors.mean(dim=(0, 1))
        elif vectors.dim() == 2:
            # Average over batch -> [vocab]
            avg_activation = vectors.mean(dim=0)
        elif vectors.dim() == 1:
            avg_activation = vectors
        else:
            # Fallback: flatten all but last dim as batch-like
            avg_activation = vectors.view(-1, vectors.shape[-1]).mean(dim=0)
        
        # The FLOPS loss is the sum of the squared mean activations (uniform weighting)
        mean_sq = torch.pow(avg_activation, 2)  # [vocab]
        flops_loss = torch.sum(mean_sq)
        
        return flops_loss


    def forward(self, candidate_news, clicked_news, clicked_news_mask):
        """
        Forward pass through the LSNR model.
        Args:
            candidate_news (list): List of candidate news dicts.
            clicked_news (list): List of clicked news dicts.
            clicked_news_mask (list): Mask for clicked news articles.
        Returns:
            click_prob (Tensor): Predicted click probabilities.
            total_reg (Tensor): Total regularization loss.
        """
        num_candidates = len(candidate_news)
        batch_size = candidate_news[0]['input_ids'].shape[0]
        # Concatenate all candidate news into a single batch
        batched_candidate = {
            'input_ids': torch.cat([news['input_ids'] for news in candidate_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in candidate_news], dim=0),
        }
        # Encode all candidate news in a single forward pass
        candidate_vecs_batched = self.news_encoder(batched_candidate)
        # Reshape: [num_candidates*batch_size, embed_dim] -> [batch_size, num_candidates, embed_dim]
        candidate_vecs = candidate_vecs_batched.view(num_candidates, batch_size, -1).transpose(0, 1)
        # Concatenate all clicked news into a single batch
        num_clicked = len(clicked_news)
        batched_clicked = {
            'input_ids': torch.cat([news['input_ids'] for news in clicked_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in clicked_news], dim=0),
        }
        # Encode all clicked news in a single forward pass
        clicked_vecs_batched = self.news_encoder(batched_clicked)
        # Reshape: [num_clicked*batch_size, embed_dim] -> [batch_size, num_clicked, embed_dim]
        clicked_vecs = clicked_vecs_batched.view(num_clicked, batch_size, -1).transpose(0, 1)
        # Apply mask to clicked news vectors
        # Support masks produced by default_collate:
        #  - list of length num_clicked with tensors of shape [batch]
        #  - list of lists [[...], ...] -> shape [batch, num_clicked]
        #  - tensor of shape [batch, num_clicked]
        if isinstance(clicked_news_mask, torch.Tensor):
            mask_bt = clicked_news_mask  # [batch, num_clicked] expected
        elif isinstance(clicked_news_mask, list):
            if len(clicked_news_mask) > 0 and isinstance(clicked_news_mask[0], torch.Tensor):
                # Transposed style: [num_clicked][batch] -> stack and transpose
                mask_bt = torch.stack(clicked_news_mask, dim=0).transpose(0, 1)
            else:
                # Assume already [batch][num_clicked]
                mask_bt = torch.tensor(clicked_news_mask)
        else:
            mask_bt = torch.tensor(clicked_news_mask)

        mask_bt = mask_bt.to(self.device, dtype=clicked_vecs.dtype)
        # Ensure shape [batch, num_clicked]
        if mask_bt.dim() == 1:
            mask_bt = mask_bt.unsqueeze(0)
        # Expand to [batch, num_clicked, 1]
        expanded_mask = mask_bt.unsqueeze(-1)
        clicked_vecs = clicked_vecs * expanded_mask
        # Get user vector
        user_vec = self.user_encoder(clicked_vecs)
        
        # Get current lambda values from scheduler
        current_lambda_n, current_lambda_u = self.lambda_scheduler.get_lambda_values()
        
        candidate_flops_reg = self._calculate_flops_regularizer(candidate_vecs) / num_candidates
        user_flops_reg = self._calculate_flops_regularizer(user_vec)

        total_reg = current_lambda_n * candidate_flops_reg + current_lambda_u * user_flops_reg
        
        # Compute click prediction
        click_prob = self.click_predictor(candidate_vecs, user_vec)
        
        # Return detailed regularization info
        reg_info = {
            'total': total_reg,
            'candidate_flops': candidate_flops_reg,
            'user_flops': user_flops_reg,
            'candidate_weighted': current_lambda_n * candidate_flops_reg,
            'user_weighted': current_lambda_u * user_flops_reg,
            'current_lambda_n': current_lambda_n,
            'current_lambda_u': current_lambda_u,
        }
        return click_prob, reg_info

    def get_news_vector(self, news):
        """
        Get encoded news vector.
        Args:
            news (dict): News input dict.
        Returns:
            Tensor: Encoded news vector.
        """
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Get encoded user vector.
        Args:
            clicked_news_vector (Tensor): Encoded clicked news vectors.
        Returns:
            Tensor: Encoded user vector.
        """
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Get click prediction from news and user vectors.
        Args:
            news_vector (Tensor): Encoded news vector.
            user_vector (Tensor): Encoded user vector.
        Returns:
            Tensor: Click probability.
        """
        return self.click_predictor(news_vector, user_vector)
    
