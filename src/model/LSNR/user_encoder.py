import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

class L1Regularizer(nn.Module):
    """
    L1 regularization term with a linearly increasing schedule for alpha.
    Encourages sparsity in user representations over training steps.
    """
    def __init__(self, T: int = 5000, alpha: float = 0.02):
        """
        Args:
            T (int): Number of steps for the regularization schedule.
            alpha (float): Maximum value for the regularization coefficient.
        """
        super().__init__()
        self.T = T
        self.max_alpha = alpha
        self.current_step = 0
        self.current_alpha = 0

    def forward(self, reps):
        """
        Compute the L1 regularization term.
        Args:
            reps (Tensor): [batch_size, ...] representation tensor.
        Returns:
            reg (Tensor): Regularization loss term.
        """
        reg = torch.abs(reps).sum(dim=1).mean() * self.current_alpha
        self.step()
        return reg

    def step(self):
        """
        Update the current step and adjust the alpha value based on the schedule.
        """
        if self.current_step < self.T:
            self.current_step += 1
            # Linear schedule for alpha
            self.current_alpha = (self.current_step / self.T) * self.max_alpha
        # After T steps, alpha stays at max_alpha

class GatedProjection(nn.Module):
    """
    Gated projection layer that computes a weighted sum of the input features using a gate and a transformation.
    Projects input features into a lower-dimensional space while maintaining important information.
    """
    def __init__(self, in_features, out_features, dropout_prob, plm):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            dropout_prob (float): Dropout probability.
            plm (nn.Module): Pre-trained language model (for initializing weights).
        """
        super().__init__()
        self.fc_gate = nn.Linear(in_features, out_features)        # Computes the gate
        self.fc_transform = nn.Linear(in_features, out_features)  # Computes the transformation
        # Use BERT's MLM output projection weights
        self.fc_transform.weight = plm.cls.predictions.decoder.weight
        self.fc_transform.bias = plm.cls.predictions.decoder.bias
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        Forward pass through the gated projection layer.
        Args:
            x (Tensor): [batch_size, in_features] input tensor.
        Returns:
            output (Tensor): [batch_size, out_features] gated output.
        """
        gate = torch.sigmoid(self.fc_gate(x))
        transform = torch.tanh(self.fc_transform(x))
        return self.dropout(gate * transform)

class UserEncoder(nn.Module):
    """
    Encodes user interactions with news articles using attention mechanisms and projection layers.
    Produces a sparse user representation for downstream tasks.
    """
    def __init__(self, config, plm, device):
        """
        Args:
            config (dict): Model configuration.
            plm (nn.Module): Pre-trained language model.
            device (torch.device): Device to run the model on.
        """
        super().__init__()
        self.config = config
        self.device = device
        self.sparse_dim = plm.config.vocab_size
        self.reduced_dim = plm.config.hidden_size
        self.embedding_dim = self.reduced_dim
        self.dropout_prob = config['dropout_probability']
        # Multi-head self-attention for temporal modeling
        self.multihead_self_attention = MultiHeadSelfAttention(self.embedding_dim, 16)
        # Additive attention for aggregation
        self.additive_attention = AdditiveAttention(config['query_vector_dim'], self.embedding_dim)
        # Dimension reduction from sparse to dense
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.sparse_dim, self.reduced_dim),
            nn.LayerNorm(self.reduced_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.reduced_dim, self.reduced_dim)
        )
        # Gated projection to sparse space
        self.gated_projection = GatedProjection(self.embedding_dim, self.sparse_dim, self.dropout_prob, plm)
        # L1 regularizer for sparsity (SPLADE-style)
        self.regularizer = L1Regularizer(alpha=0.0003, T=10000)

    def forward(self, article_vectors):
        """
        Encode user history of clicked news articles.
        Args:
            article_vectors (Tensor): [batch_size, seq_len, sparse_dim] input news vectors.
        Returns:
            user_emb (Tensor): [batch_size, sparse_dim] sparse user representation.
            reg (Tensor): Regularization loss term.
        """
        # Reduce dimension from sparse to dense
        user_emb = self.dim_reduction(article_vectors)
        # Contextualize with multi-head self-attention
        user_emb = self.multihead_self_attention(user_emb)
        # Aggregate with additive attention
        user_emb = self.additive_attention(user_emb)
        # Project to sparse space with gating
        user_emb = self.gated_projection(user_emb)
        # Apply log1p(ReLU(.)) for SPLADE-style sparsity
        user_emb = torch.log1p(torch.relu(user_emb))
        # Compute L1 regularization
        reg = self.regularizer(user_emb)
        return user_emb, reg