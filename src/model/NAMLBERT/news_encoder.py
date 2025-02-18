import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.additive import AdditiveAttention
from model.general.attention.multihead_self import MultiHeadSelfAttention
from transformers import AutoModel

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, plm, device):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.device = device

        self.plm = plm
        self.embedding_dim = self.plm.config.hidden_size

        self.category_embedding = nn.Embedding(config['num_categories'] + 1,
                                          self.embedding_dim,
                                          padding_idx=0)
    
        self.CNN = nn.Conv1d(self.embedding_dim,
                            self.embedding_dim,
                            kernel_size=3,
                            padding=1)
        
        self.additive_attention = AdditiveAttention(self.config['query_vector_dim'], 
                                                    self.embedding_dim)

        self.final_attention = AdditiveAttention(config['query_vector_dim'],
                                                     self.embedding_dim)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title,
                    "abstract": batch_size * num_words_abstract,
                    "input_ids": batch_size x 1 x seq_len,
                    "attention_mask": batch_size x 1 x seq_len
                }
        Returns:
            (shape) batch_size x embedding_dim
        """
        text = {
            "input_ids": news["input_ids"].squeeze(1).to(self.device),
            "attention_mask": news["attention_mask"].squeeze(1).to(self.device)
        }

        news_vector = self.plm(**text)
        
        news_vector = news_vector.last_hidden_state
        
        news_vector = news_vector.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        news_vector = self.CNN(news_vector)        # [batch_size, embedding_dim, seq_len]
        news_vector = news_vector.transpose(1, 2)  # [batch_size, seq_len, embedding_dim]
        
        news_vector = self.additive_attention(news_vector)

        category_vector = self.category_embedding(news['category'].to(self.device))
        subcategory_vector = self.category_embedding(news['subcategory'].to(self.device))

        final_news_vector = torch.stack([news_vector, category_vector, subcategory_vector], dim=1)
    
        final_news_vector = self.final_attention(final_news_vector)

        return final_news_vector
