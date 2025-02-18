import torch
import torch.nn as nn
from model.general.attention.additive import AdditiveAttention

class UserEncoder(nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        # The input dimension should match the news representation dimension, i.e. config['embedding_dim'].
        self.additive_attention = AdditiveAttention(
            config['query_vector_dim'],
            config['embedding_dim']
        )
    
    def forward(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: Tensor of shape (batch_size, num_clicked_news, embedding_dim)
        Returns:
            user_vector: Tensor of shape (batch_size, embedding_dim)
        """
        user_vector = self.additive_attention(clicked_news_vector)
        return user_vector 