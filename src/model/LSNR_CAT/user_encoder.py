import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

class GatedProjection(nn.Module):
    """
    Gated projection layer that computes a weighted sum of the input features using a gate and a transformation.
    This layer is used to project the input features into a lower-dimensional space while maintaining important information.
    """
    def __init__(self, in_features, out_features, dropout_prob, plm):
        """
        Gated projection layer that computes a weighted sum of the input features using a gate and a transformation.
        Args:
            in_features: number of input features
            out_features: number of output features
            dropout_prob: dropout probability
            plm: pre-trained language model (for initializing weights)
        """
        super().__init__()
        self.fc_gate = nn.Linear(in_features, out_features)        # computes the gate

        # load weights from mlm head
        self.fc_transform = nn.Linear(in_features, out_features)     # computes the transformation
        self.fc_transform.weight = plm.cls.predictions.decoder.weight
        
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        Forward pass through the gated projection layer.
        Args:
            x: input tensor of shape [batch_size, seq_len, in_features]
        Returns:
            output: gated output tensor of shape [batch_size, seq_len, out_features]
        """
        gate = torch.sigmoid(self.fc_gate(x))
        transform = torch.tanh(self.fc_transform(x))
        return self.dropout(gate * transform)

class UserEncoder(nn.Module):
    """
    UserEncoder class for processing user interactions using a pre-trained language model (PLM).
    This class applies multi-head self-attention and additive attention mechanisms to the processed user interactions.
    """
    def __init__(self, config, plm, device):
        """
        Args:
            config: configuration dictionary
            plm: pre-trained language model
            device: device to run the model on
        """
        super().__init__()
        self.config = config
        self.device = device

        self.sparse_dim = plm.config.vocab_size
        self.reduced_dim = plm.config.hidden_size
        self.embedding_dim = self.reduced_dim
        
        self.num_categories = config['num_categories'] + 1
        self.news_vector_dim = self.sparse_dim + 2 * self.embedding_dim

    def forward(self, article_vectors):
        """
        Forward pass through the UserEncoder.
        Args:
            article_vectors: tensor of shape [batch_size, num_articles, news_vector_dim]
        Returns:
            user_emb: user embedding tensor of shape [batch_size, news_vector_dim]
        """
        user_emb = torch.logsumexp(article_vectors, dim=1)
        return user_emb