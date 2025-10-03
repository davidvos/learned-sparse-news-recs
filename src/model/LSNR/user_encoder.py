import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention


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
        # Tie to the PLM's output head weights across architectures
        tied_weight = None
        tied_bias = None
        if hasattr(plm, 'cls') and hasattr(plm.cls, 'predictions') and hasattr(plm.cls.predictions, 'decoder'):
            tied_weight = plm.cls.predictions.decoder.weight
            tied_bias = getattr(plm.cls.predictions, 'bias', None)
            if tied_bias is None and hasattr(plm.cls.predictions.decoder, 'bias'):
                tied_bias = plm.cls.predictions.decoder.bias
        elif hasattr(plm, 'lm_head'):
            decoder = getattr(plm.lm_head, 'decoder', None)
            if decoder is not None and hasattr(decoder, 'weight'):
                tied_weight = decoder.weight
                tied_bias = getattr(plm.lm_head, 'bias', None)
        if tied_weight is None:
            try:
                out_emb = plm.get_output_embeddings()
                tied_weight = out_emb.weight
                tied_bias = getattr(out_emb, 'bias', None)
            except Exception:
                tied_weight = None
                tied_bias = None
        if tied_weight is not None and tied_weight.shape[0] == out_features:
            # Use PLM weights directly if dimensions match
            self.fc_transform.weight = tied_weight
            if tied_bias is not None:
                self.fc_transform.bias = tied_bias
        else:
            # Initialize with Xavier uniform if dimensions don't match
            nn.init.xavier_uniform_(self.fc_transform.weight)
            if self.fc_transform.bias is not None:
                nn.init.zeros_(self.fc_transform.bias)
            
            # Copy overlapping weights from PLM if available
            if tied_weight is not None:
                min_vocab_size = min(out_features, tied_weight.shape[0])
                with torch.no_grad():
                    self.fc_transform.weight[:min_vocab_size] = tied_weight[:min_vocab_size]
                    if tied_bias is not None and self.fc_transform.bias is not None:
                        self.fc_transform.bias[:min_vocab_size] = tied_bias[:min_vocab_size]
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
    def __init__(self, config, dataset_config, plm, device):
        """
        Args:
            config (dict): Model configuration.
            dataset_config (dict): Dataset configuration.
            plm (nn.Module): Pre-trained language model.
            device (torch.device): Device to run the model on.
        """
        super().__init__()
        self.config = config
        self.device = device
        # Use configurable sparse vocab size, fallback to PLM vocab size if not specified
        self.sparse_dim = dataset_config.get('sparse_vocab_size', plm.config.vocab_size)
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
        # Optional LayerNorm before gated projection for stability and interpretability
        self.enable_pre_gated_layernorm = bool(config.get('user_pre_gated_layernorm', False))
        if self.enable_pre_gated_layernorm:
            self.pre_gated_layernorm = nn.LayerNorm(self.embedding_dim)
        else:
            self.pre_gated_layernorm = None

        self.gated_projection = GatedProjection(self.embedding_dim, self.sparse_dim, self.dropout_prob, plm)

    def forward(self, article_vectors):
        """
        Encode user history of clicked news articles.
        Args:
            article_vectors (Tensor): [batch_size, seq_len, sparse_dim] input news vectors.
        Returns:
            user_emb (Tensor): [batch_size, sparse_dim] sparse user representation.
        """
        # Reduce dimension from sparse to dense
        user_emb = self.dim_reduction(article_vectors)
        # Contextualize with multi-head self-attention
        user_emb = self.multihead_self_attention(user_emb)
        # Aggregate with additive attention
        user_emb = self.additive_attention(user_emb)
        # Project to sparse space with gating
        if self.pre_gated_layernorm is not None:
            user_emb = self.pre_gated_layernorm(user_emb)
        user_emb = self.gated_projection(user_emb)
        # Apply log1p(ReLU(.)) for SPLADE-style sparsity
        user_emb = torch.log1p(torch.relu(user_emb))
        return user_emb