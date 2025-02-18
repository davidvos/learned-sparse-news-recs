import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.additive import AdditiveAttention

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, plm, device):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.device = device
        
        self.plm = plm
        self.embedding_dim = self.plm.config.hidden_size

        self.category_embedding = nn.Embedding(config['num_categories'], self.embedding_dim, padding_idx=0)
        
        # Single additive attention for title encoding
        self.additive_attention = AdditiveAttention(self.config['query_vector_dim'], 
                                                  self.embedding_dim)
        
        # Linear layer to combine all features
        self.linear = nn.Linear(self.embedding_dim * 3, self.embedding_dim)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, embedding_dim
        """
        text = {
            "input_ids": news["input_ids"].squeeze(1).to(self.device),
            "attention_mask": news["attention_mask"].squeeze(1).to(self.device)
        }

        # Get BERT embeddings
        news_vector = self.plm(**text).last_hidden_state

        # Apply dropout
        news_vector = F.dropout(news_vector,
                              p=self.config['dropout_probability'],
                              training=self.training)
        
        # Get attended title representation
        title_vector = self.additive_attention(news_vector)
        
        # Get category embeddings
        category_vector = self.category_embedding(news["category"].to(self.device))
        subcategory_vector = self.category_embedding(news["subcategory"].to(self.device))
        
        # Concatenate all features
        combined_vector = torch.cat([title_vector, category_vector, subcategory_vector], dim=1)
        
        # Project back to embedding_dim
        final_vector = self.linear(combined_vector)

        return final_vector
