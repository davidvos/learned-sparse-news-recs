import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

class L1Regularizer(nn.Module):
    """
    L1 regularization term that increases over time.
    This regularization term is used to encourage sparsity in user representations.
    """
    def __init__(self, T: int = 5000, alpha: float = 0.02):
        """
        Initialize the L1 regularizer with a schedule for alpha.
        Args:
            T: number of steps for the regularization schedule
            alpha: maximum value for the regularization term
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
            reps: tensor of shape [batch_size, seq_len, embed_dim]
        Returns:
            reg: regularization term
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
            self.current_alpha = (self.current_step / self.T) ** 2 * self.max_alpha
        else:
            pass

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
        
        # Increase number of attention heads for better temporal modeling
        self.multihead_self_attention = MultiHeadSelfAttention(
            self.embedding_dim, 16)
        
        self.additive_attention = AdditiveAttention(
            config['query_vector_dim'],
            self.embedding_dim
        ) 
        
        # Improved dimension reduction with gated mechanism
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.sparse_dim, self.reduced_dim),
            nn.LayerNorm(self.reduced_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.reduced_dim, self.reduced_dim)
        )
         
        self.gated_projection = GatedProjection(self.embedding_dim, self.sparse_dim, self.dropout_prob, plm)
        
        # Weaker regularization for user embeddings
        self.regularizer = L1Regularizer(alpha=0.004, T=10000)

    def forward(self, article_vectors):
        """
        Forward pass through the UserEncoder.
        Args:
            article_vectors: tensor of shape [batch_size, num_articles, sparse_dim]
        Returns:
            user_emb: user embedding tensor of shape [batch_size, reduced_dim]
            reg: regularization term
        """
        user_emb = self.dim_reduction(article_vectors)
        
        user_emb = self.multihead_self_attention(user_emb)

        user_emb = self.additive_attention(user_emb)
            
        user_emb = self.gated_projection(user_emb)

        user_emb = torch.log1p(F.relu(user_emb)) 

        reg = self.regularizer(user_emb)
        
        return user_emb, reg