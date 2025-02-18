import torch
from model.general.attention.additive import AdditiveAttention

class UserEncoder(torch.nn.Module):
    def __init__(self, config, plm):
        super(UserEncoder, self).__init__()

        self.embedding_dim = plm.config.hidden_size 
        self.additive_attention = AdditiveAttention(config['query_vector_dim'],
                                                    self.embedding_dim)

    def forward(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        user_vector = self.additive_attention(clicked_news_vector)
        return user_vector
