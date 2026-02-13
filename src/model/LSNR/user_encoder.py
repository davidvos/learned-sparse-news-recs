import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import MultiHeadSelfAttention, AdditiveAttention

class UserEncoder(nn.Module):
    """
    Encodes user history using DENSE news vectors and outputs a SPARSE user vector.
    """
    def __init__(self, config, dataset_config, plm, device):
        super().__init__()
        self.device = device
        self.hidden_size = plm.config.hidden_size
        self.sparse_dim = dataset_config.get('sparse_vocab_size', plm.config.vocab_size)
        self.dropout_prob = config.get('dropout_probability', 0.1)
        self.max_history_len = config.get('max_history_len', 50)
        
        num_heads = config.get('num_attention_heads', 16)
        self.multihead_self_attention = MultiHeadSelfAttention(self.hidden_size, n_head=num_heads, device=self.device)
        self.attn_W = nn.Linear(self.hidden_size, config.get('query_vector_dim', 200))
        self.attn_v = nn.Linear(config.get('query_vector_dim', 200), 1, bias=False)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Always use asymmetric projection (independent from NewsEncoder)
        self.projection = nn.Linear(self.hidden_size, self.sparse_dim, bias=False)
        self._init_projection_from_mlm(plm)
    
    def _init_projection_from_mlm(self, plm):
        if self.sparse_dim == plm.config.vocab_size:
            with torch.no_grad():
                self.projection.weight.copy_(plm.cls.predictions.decoder.weight)
        else:
            nn.init.xavier_uniform_(self.projection.weight)
            min_vocab = min(self.sparse_dim, plm.config.vocab_size)
            with torch.no_grad():
                self.projection.weight[:min_vocab].copy_(plm.cls.predictions.decoder.weight[:min_vocab])

    def forward(self, dense_article_vectors, sparse_article_vectors, mask=None):
        x = dense_article_vectors
        residual = x
        if mask is not None:
            click_length = mask.sum(dim=1).to(self.device).long()
            click_length = click_length.clamp(min=1) 
            x = self.multihead_self_attention(x, length=click_length)
        else:
            x = self.multihead_self_attention(x)
        
        x = x + residual
        x = self.layer_norm(x)

        projected = torch.tanh(self.attn_W(self.dropout(x)))
        scores = self.attn_v(projected)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=1)
        user_emb_sparse = torch.bmm(attn_weights.transpose(1, 2), sparse_article_vectors).squeeze(1)

        return user_emb_sparse