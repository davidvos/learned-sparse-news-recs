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

# NEW: GatedProjection module for the output projection
class GatedProjection(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob, plm):
        super().__init__()
        self.fc_gate = nn.Linear(in_features, out_features)        # computes the gate

        # load weights from mlm head
        self.fc_transform = nn.Linear(in_features, out_features)     # computes the transformation
        self.fc_transform.weight = plm.cls.predictions.decoder.weight
        
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Compute gated output: dropout(sigmoid(gate) * tanh(transform))
        gate = torch.sigmoid(self.fc_gate(x))
        transform = torch.tanh(self.fc_transform(x))
        return self.dropout(gate * transform)

class UserEncoder(nn.Module):
    def __init__(self, config, plm, device):
        super().__init__()
        self.config = config
        self.device = device

        self.sparse_dim = plm.config.vocab_size
        self.reduced_dim = plm.config.hidden_size
        self.embedding_dim = self.reduced_dim
        
        # Use config dropout probability
        self.dropout_prob = config['dropout_probability']
        
        # Remove custom multi-head self attention and instead use stacked Transformer Encoder layers:
        self.transformer_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=8, 
                dropout=self.dropout_prob, 
                activation='gelu'
            ),
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=8, 
                dropout=self.dropout_prob, 
                activation='gelu'
            )
        ])
        
        # Keep the additive attention for aggregating the sequence into a single user context vector
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
        
        # Dimension reduction
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.sparse_dim, self.reduced_dim),
            nn.LayerNorm(self.reduced_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.reduced_dim, self.reduced_dim)
        )
        
        # Remove the original output_projection and use the new gated projection instead.
        self.gated_projection = GatedProjection(self.embedding_dim, self.sparse_dim, self.dropout_prob, plm)
        
        self.layer_norm1 = nn.LayerNorm(self.reduced_dim)
        # Removed layer_norm2 since the Transformer layers perform internal normalization
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Weaker regularization for user embeddings
        self.regularizer = L1Regularizer(alpha=0.004, T=5000)

    def forward(self, article_vectors):
        # Apply temporal decay
        seq_len = article_vectors.size(1)
        time_weights = self.time_decay[:seq_len].to(self.device)
        article_vectors = article_vectors * time_weights[None, :, None]
        
        # Dimension reduction
        reduced_articles = self.dim_reduction(article_vectors)
        reduced_articles = self.dropout(reduced_articles)
        
        # Add position embeddings and normalize
        reduced_articles = reduced_articles + self.position_embedding[:, :reduced_articles.size(1), :]
        reduced_articles = self.layer_norm1(reduced_articles)
        
        # Apply the stacked transformer encoder layers (note the transpose to match (seq_len, batch, embedding))
        x = reduced_articles.transpose(0, 1)  # now shape: (seq_len, batch, embedding_dim)
        for layer in self.transformer_encoder_layers:
            x = layer(x)
        x = x.transpose(0, 1)  # back to (batch, seq_len, embedding_dim)
        
        # Aggregate the sequence with additive attention
        user_context = self.additive_attention(x)
        
        # Use the gated projection followed by softplus for the final user embedding
        user_emb = self.gated_projection(user_context)
        user_emb = F.softplus(user_emb)
        
        # Apply regularization
        reg = self.regularizer(user_emb)
        
        return user_emb, reg