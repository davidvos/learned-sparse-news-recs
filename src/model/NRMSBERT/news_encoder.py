import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, plm, device):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.device = device

        self.plm = plm

        self.embedding_dim = self.plm.config.hidden_size

        self.multihead_self_attention = MultiHeadSelfAttention(
            self.embedding_dim, config['num_attention_heads'])
        
        self.additive_attention = AdditiveAttention(config['query_vector_dim'],
                                                    self.embedding_dim)                                       

    def forward(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        news = {
            "input_ids": news["input_ids"].squeeze(1).to(self.device),
            "attention_mask": news["attention_mask"].squeeze(1).to(self.device)
        }
        
        news_vector = self.plm(**news)
        
        news_vector = news_vector.last_hidden_state
                        
        news_vector = self.multihead_self_attention(news_vector)
                        
        news_vector = self.additive_attention(news_vector)

        return news_vector