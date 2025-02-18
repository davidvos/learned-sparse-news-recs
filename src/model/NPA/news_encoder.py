import torch
import torch.nn as nn
from model.general.attention.additive import AdditiveAttention

class NewsEncoder(nn.Module):
    def __init__(self, config, plm, device):
        super(NewsEncoder, self).__init__()
        self.device = device
        self.plm = plm
        self.embedding_dim = self.plm.config.hidden_size
        # (Word embeddings are handled by the PLM, so we remove the separate embedding layer)
        # CNN to capture local patterns from the text representations.
        self.CNN = nn.Conv1d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=3,
            padding=1
        )
        # Additive attention to pool the CNN features (over the text tokens).
        self.additive_attention = AdditiveAttention(
            config['query_vector_dim'],
            self.embedding_dim
        )
        # Final attention to combine the text representation with category info.
        self.final_attention = AdditiveAttention(
            config['query_vector_dim'],
            self.embedding_dim
        )
        # Embedding for category (and subcategory – reusing the same embedding as in NAML).
        self.category_embedding = nn.Embedding(
            config['num_categories'] + 1,
            self.embedding_dim,
            padding_idx=0
        )
    
    def forward(self, news):
        """
        Args:
            news (dict): A dictionary with keys:
                "title": tensor of shape (batch_size, num_words) containing token indices.
                "category": tensor of shape (batch_size) with category ids.
                "subcategory": tensor of shape (batch_size) with subcategory ids.
        Returns:
            final_news_vector (tensor): Tensor of shape (batch_size, embedding_dim)
        """
        # --- Encode the text using a pretrained language model ---
        text = {
            "input_ids": news["input_ids"].squeeze(1).to(self.device),
            "attention_mask": news["attention_mask"].squeeze(1).to(self.device)
        }
        news_vector = self.plm(**text)
        news_vector = news_vector.last_hidden_state  # (batch_size, seq_len, embedding_dim)
        # Apply CNN over the token representations
        news_vector = news_vector.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        news_vector = self.CNN(news_vector)        # (batch_size, embedding_dim, seq_len)
        news_vector = news_vector.transpose(1, 2)  # (batch_size, seq_len, embedding_dim)
        # Apply additive attention to pool token representations into a single vector.
        news_vector = self.additive_attention(news_vector)  # (batch_size, embedding_dim)

        # --- Process category and subcategory ---
        category_vector = self.category_embedding(news["category"].to(self.device))       # (batch_size, embedding_dim)
        subcategory_vector = self.category_embedding(news["subcategory"].to(self.device))   # (batch_size, embedding_dim)

        # --- Combine text and category representations ---
        # Stack the news vector (from text) with the category and subcategory vectors, then fuse.
        stacked = torch.stack([news_vector, category_vector, subcategory_vector], dim=1)  # (batch_size, 3, embedding_dim)
        final_news_vector = self.final_attention(stacked)  # (batch_size, embedding_dim)

        return final_news_vector 