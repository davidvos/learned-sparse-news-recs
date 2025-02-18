import torch
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

class UserEncoder(torch.nn.Module):
    def __init__(self, config, plm, lm_head, device):
        super(UserEncoder, self).__init__()
        self.config = config
        self.device = device

        self.embedding_dim = plm.config.hidden_size

        self.lm_head = lm_head

        self.multihead_self_attention = MultiHeadSelfAttention(
            self.embedding_dim, config['num_attention_heads'])
        
        self.additive_attention = AdditiveAttention(config['query_vector_dim'],
                                                    self.embedding_dim) 

    def forward(self, user_vector):
        """
        Args:
            user_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """

        user_vector = F.dropout(user_vector,
                                p=self.config['dropout_probability'],
                                training=self.training)
        
        user_vector = self.multihead_self_attention(user_vector)
        
        user_vector = F.dropout(user_vector,
                                p=self.config['dropout_probability'],
                                training=self.training)
        
        # user_vector = self.additive_attention(user_vector)
        
        logits = self.lm_head(user_vector)

        expansion_scores = torch.log1p(1 + torch.relu(logits))
        final_user_vector = torch.max(expansion_scores, dim=1).values

        return final_user_vector
