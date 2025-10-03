import torch
import torch.nn as nn
from model.LSTURBERT.news_encoder import NewsEncoder
from model.LSTURBERT.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor
from transformers import AutoModel

class LSTURBERT(torch.nn.Module):
    """
    NAML network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, model_config, dataset_config, device):
        super(LSTURBERT, self).__init__()
        self.model_config = model_config
        self.dataset_config = dataset_config

        self.plm = AutoModel.from_pretrained(dataset_config['pretrained_model_name'])
        self.embedding_dim = self.plm.config.hidden_size
        self.device = device

        self.news_encoder = NewsEncoder(model_config, self.plm, device)
        self.user_encoder = UserEncoder(model_config, self.plm)
        self.click_predictor = DotProductClickPredictor()

        self.user_embedding = nn.Embedding(model_config['num_users'] + 1, self.embedding_dim, padding_idx=0)

    def forward(self, candidate_news, clicked_news, clicked_news_mask, user):
        """
        Args:
            candidate_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title,
                        "abstract": batch_size * num_words_abstract
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": batch_size,
                        "subcategory": batch_size,
                        "title": batch_size * num_words_title,
                        "abstract": batch_size * num_words_abstract
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size
        """
        # Batch process candidate news: concatenate along batch dim and encode once
        num_candidates = len(candidate_news)
        batch_size = candidate_news[0]['input_ids'].shape[0]
        batched_candidate = {
            'input_ids': torch.cat([news['input_ids'] for news in candidate_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in candidate_news], dim=0),
            'category': torch.cat([news['category'] for news in candidate_news], dim=0),
            'subcategory': torch.cat([news['subcategory'] for news in candidate_news], dim=0),
        }
        candidate_vecs_batched = self.news_encoder(batched_candidate)
        # Reshape back to (batch_size, num_candidates, embedding_dim)
        candidate_news_vector = candidate_vecs_batched.view(num_candidates, batch_size, -1).transpose(0, 1)

        # Batch process clicked news similarly
        num_clicked = len(clicked_news)
        batched_clicked = {
            'input_ids': torch.cat([news['input_ids'] for news in clicked_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in clicked_news], dim=0),
            'category': torch.cat([news['category'] for news in clicked_news], dim=0),
            'subcategory': torch.cat([news['subcategory'] for news in clicked_news], dim=0),
        }
        clicked_vecs_batched = self.news_encoder(batched_clicked)
        clicked_news_vector = clicked_vecs_batched.view(num_clicked, batch_size, -1).transpose(0, 1)
        
        # Apply mask to clicked news vectors
        clicked_news_mask_tensor = torch.stack(clicked_news_mask).to(self.device)
        clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_news_vector = clicked_news_vector * expanded_mask

        user = self.user_embedding(user.to(self.device))

        user_vector = self.user_encoder(user, clicked_news_vector)

        click_probability = self.click_predictor(candidate_news_vector, user_vector)

        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title,
                    "abstract": batch_size * num_words_abstract
                }
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        return self.news_encoder(news)

    def get_user_vector(self, user, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_filters
        user = self.user_embedding(user.to(self.device))
        return self.user_encoder(user, clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        return self.click_predictor(news_vector, user_vector)
    
