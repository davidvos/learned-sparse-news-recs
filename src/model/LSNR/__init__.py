import torch
from transformers import AutoModelForMaskedLM
from model.LSNR.news_encoder import NewsEncoder
from model.LSNR.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor
import torch.nn as nn

class LSNR(torch.nn.Module):
    """
    LSNR class for processing news articles and user interactions using a pre-trained language model (PLM).
    """
    def __init__(self, model_config, dataset_config, device):
        """
        Initialize the LSNR model with a pre-trained language model (PLM) and device.
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
        Forward pass through the LSNR model.
        Args:
            candidate_news: list of candidate news articles
            clicked_news: list of clicked news articles
            clicked_news_mask: mask for clicked news articles
        Returns:
            click_prob: predicted click probabilities
            total_reg: total regularization term
        """
        # Batch process candidate news
        num_candidates = len(candidate_news)
        batch_size = candidate_news[0]['input_ids'].shape[0]
        
        # Concatenate all candidate news into a single batch
        batched_candidate = {
            'input_ids': torch.cat([news['input_ids'] for news in candidate_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in candidate_news], dim=0),
        }
        
        # Process all candidate news in a single forward pass
        candidate_vecs_batched, candidate_reg_batched = self.news_encoder(batched_candidate)
        
        # Reshape results back to original structure: [num_candidates*batch_size, embed_dim] -> [batch_size, num_candidates, embed_dim]
        candidate_vecs = candidate_vecs_batched.view(num_candidates, batch_size, -1).transpose(0, 1)
        
        # Process regularization
        news_reg = torch.mean(candidate_reg_batched)
        
        # Batch process clicked news
        num_clicked = len(clicked_news)
        
        # Concatenate all clicked news into a single batch
        batched_clicked = {
            'input_ids': torch.cat([news['input_ids'] for news in clicked_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in clicked_news], dim=0),
        }
        
        # Process all clicked news in a single forward pass
        clicked_vecs_batched, _ = self.news_encoder(batched_clicked)
        
        # Reshape results: [num_clicked*batch_size, embed_dim] -> [batch_size, num_clicked, embed_dim]
        clicked_vecs = clicked_vecs_batched.view(num_clicked, batch_size, -1).transpose(0, 1)
        
        # Apply mask to clicked news vectors
        clicked_news_mask_tensor = torch.stack(clicked_news_mask).to(self.device)
        clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_vecs = clicked_vecs * expanded_mask
        
        # Get user vector and user regularization
        user_vec, user_reg = self.user_encoder(clicked_vecs)
        
        # Calculate total regularization
        total_reg = (user_reg + news_reg)
        
        # Compute click prediction
        click_prob = self.click_predictor(candidate_vecs, user_vec)
        
        return click_prob, total_reg

    def get_news_vector(self, news):
       
        return self.news_encoder(news)[0]

    def get_user_vector(self, clicked_news_vector):
        
        return self.user_encoder(clicked_news_vector)[0]

    def get_prediction(self, news_vector, user_vector):
        
        return self.click_predictor(news_vector, user_vector)