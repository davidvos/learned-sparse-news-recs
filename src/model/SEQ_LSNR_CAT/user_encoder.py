import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SEQ_LSNR.news_encoder import NewsEncoder
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention
import math

class L1Regularizer(nn.Module):
    def __init__(self, T: int = 5000, alpha: float = 0.02):
        """
        Parameters
        ---------
        T: int 
            number of warming up steps
        alpha: float 
            regularization weight
        """
        super().__init__()
        self.T = T
        self.max_alpha = alpha
        self.current_step = 0
        self.current_alpha = 0

    def forward(self, reps):
        """
        Calculate L1 for an input tensor and perform a warming up step for the regularization weight
        Parameters
        ----------
        reps: torch.Tensor
            Two dimensional input Tensor
        Returns
        -------
        torch.Tensor:
            The result of L1 applied in the input tensor. 
            L1(reps) = current_alpha * mean(L1(reps_i))
        """
        reg = torch.abs(reps).sum(dim=1).mean() * self.current_alpha
        self.step()
        return reg

    def step(self):
        """
        Perform a warming up step. This warming up step would allow us to apply the regularization gradually,
        step-by-step with increasing weight over time. Without this warming up, the loss might be overwhelmed by
        the regularization right from the start, leading to the issue that the model produces very sparse but
        not semantically useful vectors in the end.
        Parameters
        ----------
        This method does not have any input parameters
        Returns
        -------
        This method does not return anything
        """
        if self.current_step < self.T:
            self.current_step += 1
            self.current_alpha = (self.current_step / self.T) ** 2 * self.max_alpha
        else:
            pass

class UserEncoder(nn.Module):
    def __init__(self, config, plm, device):
        super().__init__()
        self.config = config
        self.device = device

        self.sparse_dim = plm.config.vocab_size
        self.reduced_dim = plm.config.hidden_size
        self.embedding_dim = self.reduced_dim
        
        self.num_categories = config['num_categories'] + 1

        # Use config dropout probability
        self.dropout_prob = config['dropout_probability']
        
        # Increase number of attention heads for better temporal modeling
        self.multihead_self_attention = MultiHeadSelfAttention(
            self.embedding_dim, 16)
        
        self.additive_attention = AdditiveAttention(
            config['query_vector_dim'],
            self.embedding_dim
        ) 
        
        # Add time decay for temporal attention
        max_history = config.get('max_history_length', 50)
        self.time_decay = nn.Parameter(
            torch.exp(-torch.arange(max_history) / 10.0),  # exponential decay
            requires_grad=True
        )
        
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_history, self.embedding_dim) * 0.02  # smaller init
        )
        
        # Improved dimension reduction with gated mechanism
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.sparse_dim, self.reduced_dim),
            nn.LayerNorm(self.reduced_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.reduced_dim, self.reduced_dim)
        )
        
        # Add gating mechanism for output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.reduced_dim, self.reduced_dim),
            nn.LayerNorm(self.reduced_dim),
            nn.GELU(),
            
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.reduced_dim, self.sparse_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(self.reduced_dim)
        self.layer_norm2 = nn.LayerNorm(self.reduced_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Weaker regularization for user embeddings
        self.regularizer = L1Regularizer(alpha=0.0004, T=5000)

    def forward(self, article_vectors):
        # Apply temporal decay
        seq_len = article_vectors.size(1)

        article_vectors = article_vectors[:, :, :-self.num_categories]
        article_categories = article_vectors[:, :, -self.num_categories:]

        time_weights = self.time_decay[:seq_len].to(self.device)
        article_vectors = article_vectors * time_weights[None, :, None]
        
        # Dimension reduction
        reduced_articles = self.dim_reduction(article_vectors)
        reduced_articles = self.dropout(reduced_articles)
        
        # Add position embeddings
        reduced_articles = reduced_articles + self.position_embedding[:, :reduced_articles.size(1), :]
        reduced_articles = self.layer_norm1(reduced_articles)
        
        # Self attention with residual
        attended = self.multihead_self_attention(reduced_articles)
        attended = reduced_articles + self.dropout(attended)
        attended = self.layer_norm2(attended)
        
        # User embedding with additive attention
        user_emb = self.additive_attention(attended)
        
        # Gated output projection
        user_emb = self.output_projection(user_emb)
        user_emb = torch.log1p(F.relu(user_emb))  # [batch, seq_len, vocab_size]

        # sum article_categories over the sequence length and normalize and then concat to user_emb
        article_categories = article_categories.sum(dim=1)
        # article_categories = article_categories / article_categories.sum(dim=1, keepdim=True)
        user_emb = torch.cat([user_emb, article_categories], dim=1)
        
        # Apply regularization
        reg = self.regularizer(user_emb)
        
        return user_emb, reg