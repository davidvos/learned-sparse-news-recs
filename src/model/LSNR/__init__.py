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
            dataset_config['pretrained_model_name'], output_hidden_states=True)
        self.device = device
        # News and user encoders
        self.news_encoder = NewsEncoder(model_config, self.plm, device)
        self.user_encoder = UserEncoder(model_config, self.plm, device)
        # Click predictor (dot product)
        self.click_predictor = DotProductClickPredictor()

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
        candidate_vecs_batched, candidate_reg_batched = self.news_encoder(batched_candidate)
        # Reshape: [num_candidates*batch_size, embed_dim] -> [batch_size, num_candidates, embed_dim]
        candidate_vecs = candidate_vecs_batched.view(num_candidates, batch_size, -1).transpose(0, 1)
        # Aggregate regularization
        news_reg = torch.mean(candidate_reg_batched)
        # Concatenate all clicked news into a single batch
        num_clicked = len(clicked_news)
        batched_clicked = {
            'input_ids': torch.cat([news['input_ids'] for news in clicked_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in clicked_news], dim=0),
        }
        # Encode all clicked news in a single forward pass
        clicked_vecs_batched, _ = self.news_encoder(batched_clicked)
        # Reshape: [num_clicked*batch_size, embed_dim] -> [batch_size, num_clicked, embed_dim]
        clicked_vecs = clicked_vecs_batched.view(num_clicked, batch_size, -1).transpose(0, 1)
        # Apply mask to clicked news vectors
        clicked_news_mask_tensor = torch.stack(clicked_news_mask).to(self.device)
        clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_vecs = clicked_vecs * expanded_mask
        # Get user vector and user regularization
        user_vec, user_reg = self.user_encoder(clicked_vecs)
        # Total regularization
        total_reg = user_reg + news_reg
        # Compute click prediction
        click_prob = self.click_predictor(candidate_vecs, user_vec)
        return click_prob, total_reg

    def get_news_vector(self, news):
        """
        Get encoded news vector.
        Args:
            news (dict): News input dict.
        Returns:
            Tensor: Encoded news vector.
        """
        return self.news_encoder(news)[0]

    def get_user_vector(self, clicked_news_vector):
        """
        Get encoded user vector.
        Args:
            clicked_news_vector (Tensor): Encoded clicked news vectors.
        Returns:
            Tensor: Encoded user vector.
        """
        return self.user_encoder(clicked_news_vector)[0]

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