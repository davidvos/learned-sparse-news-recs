import torch
from transformers import AutoModelForMaskedLM
from model.LSNR_CAT.news_encoder import NewsEncoder
from model.LSNR_CAT.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor
import torch.nn as nn

class LSNR_CAT(torch.nn.Module):
    """
    LSNR_CAT class for processing news articles and user interactions using a pre-trained language model (PLM).
    """
    def __init__(self, model_config, dataset_config, device):
        """
        Initialize the LSNR_CAT model with a pre-trained language model (PLM) and device.
        Args:
            model_config: model-specific configuration dictionary
            dataset_config: dataset-specific configuration dictionary
            device: device to run the model on
        """
        super().__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.plm = AutoModelForMaskedLM.from_pretrained(dataset_config['pretrained_model_name'], output_hidden_states=True)
        
        self.device = device

        self.news_encoder = NewsEncoder(model_config, self.plm, device)
        self.user_encoder = UserEncoder(model_config, self.plm, device)
        self.click_predictor = DotProductClickPredictor()
        
    def forward(self, candidate_news, clicked_news, clicked_news_mask):
        """
        Forward pass through the LSNR_CAT model.
        Args:
            candidate_news: list of candidate news articles
            clicked_news: list of clicked news articles
            clicked_news_mask: mask for clicked news articles
        Returns:
            click_prob: predicted click probabilities
            regularization: regularization term (always 0 for LSNR_CAT)
        """
        num_candidates = len(candidate_news)
        batch_size = candidate_news[0]['input_ids'].shape[0]
        
        # Concatenate all candidate news into a single batch
        batched_candidate = {
            'input_ids': torch.cat([news['input_ids'] for news in candidate_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in candidate_news], dim=0),
            'category': torch.cat([news['category'] for news in candidate_news], dim=0),
            'subcategory': torch.cat([news['subcategory'] for news in candidate_news], dim=0),
        }
        
        # Process all candidate news in a single forward pass
        candidate_vecs_batched = self.news_encoder(batched_candidate)
        
        # Reshape results back to original structure: [num_candidates*batch_size, embed_dim] -> [batch_size, num_candidates, embed_dim]
        candidate_vecs = candidate_vecs_batched.view(num_candidates, batch_size, -1).transpose(0, 1)
        
        # Batch process clicked news
        num_clicked = len(clicked_news)
        
        # Concatenate all clicked news into a single batch
        batched_clicked = {
            'input_ids': torch.cat([news['input_ids'] for news in clicked_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in clicked_news], dim=0),
            'category': torch.cat([news['category'] for news in clicked_news], dim=0),
            'subcategory': torch.cat([news['subcategory'] for news in clicked_news], dim=0),
        }
        
        # Process all clicked news in a single forward pass
        clicked_vecs_batched = self.news_encoder(batched_clicked)
        
        # Reshape results: [num_clicked*batch_size, embed_dim] -> [batch_size, num_clicked, embed_dim]
        clicked_vecs = clicked_vecs_batched.view(num_clicked, batch_size, -1).transpose(0, 1)
        
        # Apply mask to clicked news vectors
        clicked_news_mask_tensor = torch.stack(clicked_news_mask).to(self.device)
        clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_vecs = clicked_vecs * expanded_mask
        
        # Get user vector
        user_vec = self.user_encoder(clicked_vecs)
        
        # Compute click prediction
        click_prob = self.click_predictor(candidate_vecs, user_vec)
        
        regularization = 0
        return click_prob, regularization
    
    def get_news_vector(self, news):
        """
        Get the vector representation of a news article.
        Args:
            news: dictionary containing 'input_ids' and 'attention_mask'
        Returns:
            news_vector: processed news representation
        """
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Get the vector representation of clicked news articles.
        Args:
            clicked_news_vector: tensor of shape [batch_size, num_clicked, embed_dim]
        Returns:
            user_vector: processed user representation
        """
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        return self.click_predictor(news_vector, user_vector)