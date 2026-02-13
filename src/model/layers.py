import torch
import torch.nn as nn
import torch.nn.functional as F

class DotProductClickPredictor(nn.Module):
    def __init__(self, use_bm25_saturation=False, bm25_k1=1.2):
        super(DotProductClickPredictor, self).__init__()
        self.use_bm25_saturation = use_bm25_saturation
        self.bm25_k1 = bm25_k1

    def _apply_bm25_saturation(self, vec: torch.Tensor) -> torch.Tensor:
        """Apply BM25-style saturation: tf / (k1 + tf)"""
        return vec / (self.bm25_k1 + vec + 1e-8)

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, candidate_size, X (sparse SPLADE-style vectors)
            user_vector: batch_size, X (sparse SPLADE-style vectors)
        Returns:
            (shape): batch_size, candidate_size
        """
        # Apply BM25 saturation symmetrically to both sides if enabled
        if self.use_bm25_saturation:
            candidate_news_vector = self._apply_bm25_saturation(candidate_news_vector)
            user_vector = self._apply_bm25_saturation(user_vector)
        
        # Compute dot product
        # batch_size, candidate_size
        probability = torch.bmm(candidate_news_vector, user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        return probability


class AdditiveAttention(nn.Module):
    """
    A general additive attention module.
    Originally for NAML.
    """
    def __init__(self,
                 query_vector_dim,
                 candidate_vector_dim,
                 writer=None,
                 tag=None,
                 names=None):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector, mask=None, return_weights=False):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
            mask: batch_size, candidate_size (1=Real, 0=Padding)
            return_weights: if True, returns (aggregation, weights)
        Returns:
            (shape) batch_size, candidate_vector_dim (if return_weights=False)
            (shape) batch_size, candidate_size (if return_weights=True)
        """
        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(self.linear(candidate_vector))
        
        # batch_size, candidate_size
        scores = torch.matmul(temp, self.attention_query_vector)
        
        if mask is not None:
            # Set scores of padded items to -inf before softmax
            scores = scores.masked_fill(mask == 0, -1e9)
            
        candidate_weights = F.softmax(scores, dim=1)
       
        if return_weights:
            return candidate_weights

        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module.
    """
    def __init__(self, d_model, n_head, dropout=0.1, device=None):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, length=None):
        # x: [batch, seq, d_model]
        batch_size, seq_len, _ = x.size()
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # scores: [batch, n_head, seq, seq]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if length is not None:
            # Create mask [batch, seq] -> [batch, 1, 1, seq]
            mask = torch.arange(seq_len, device=x.device)[None, :] < length[:, None]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V) # [batch, n_head, seq, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(out)

class SimilarityAttention(nn.Module):
    """
    A general attention module based on similarity w.r.t. another vector
    """
    def __init__(self):
        super(SimilarityAttention, self).__init__()

    def forward(self, wrt_vector, candidate_vector):
        """
        Args:
            wrt_vector: batch_size, candidate_vector_dim
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.bmm(
            candidate_vector, wrt_vector.unsqueeze(dim=2)).squeeze(dim=2),
                                      dim=1)
        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target

