"""LSNR User Encoder.

Aggregates the user's history of interacted news articles into a single
sparse vector. Supports pathway ablations ('sparse', 'dense', 'both') and
multiple aggregation strategies (attention, average, decay).
"""
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
        query_vector_dim = config.get('query_vector_dim', 200)

        num_heads = config.get('num_attention_heads', 16)
        self.multihead_self_attention = MultiHeadSelfAttention(self.hidden_size, n_head=num_heads, device=self.device)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.additive_attention = AdditiveAttention(query_vector_dim, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        # Attention over history for sparse aggregation
        self.attn_W = nn.Linear(self.hidden_size, query_vector_dim)
        self.attn_v = nn.Linear(query_vector_dim, 1, bias=False)

        # SPLADE-style projection from dense user to sparse space
        self.projection = nn.Linear(self.hidden_size, self.sparse_dim, bias=False)
        self.topk_sparsity_queries = config.get('topk_sparsity_queries', None)
        self.user_aggregation = config.get('user_aggregation', 'attention')
        self.temporal_decay_beta = float(config.get('temporal_decay_beta', 0.9))
        self.use_position_embeddings = bool(config.get('use_position_embeddings', False))
        self.user_pathway = config.get('user_pathway', 'both')
        if self.use_position_embeddings:
            self.max_position_embeddings = int(dataset_config.get('num_clicked_news_a_user', 50))
            self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        self._init_projection_weights(plm)

    def _init_projection_weights(self, plm):
        if self.sparse_dim == plm.config.vocab_size:
            # Tie weights directly to the input word embeddings
            self.projection.weight = plm.get_input_embeddings().weight
        else:
            # Copy sliced input word embeddings if sparse_dim differs
            with torch.no_grad():
                sliced_weight = plm.get_input_embeddings().weight[:self.sparse_dim].clone()
                self.projection.weight.copy_(sliced_weight)

    def forward(self, dense_article_vectors, sparse_article_vectors, mask=None, return_proj=False, return_dense=False):
        x = dense_article_vectors

        # Add right-aligned learned positional embeddings if enabled
        if self.use_position_embeddings:
            if mask is not None:
                history_lengths = mask.sum(dim=1, keepdim=True).long()
                seq_len = x.size(1)
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
                pos_idx = (self.max_position_embeddings - 1) - ((history_lengths - 1) - positions)
                pos_idx = torch.clamp(pos_idx, 0, self.max_position_embeddings - 1)
                pos_idx = pos_idx * mask.long()
                pos_emb = self.position_embeddings(pos_idx) * mask.unsqueeze(-1)
            else:
                seq_len = x.size(1)
                positions = torch.arange(seq_len, device=x.device)
                pos_idx = (self.max_position_embeddings - 1) - ((seq_len - 1) - positions)
                pos_idx = torch.clamp(pos_idx, 0, self.max_position_embeddings - 1)
                pos_emb = self.position_embeddings(pos_idx).unsqueeze(0)
            x = x + pos_emb

        residual = x

        if mask is not None:
            click_length = mask.sum(dim=1).to(self.device).long().clamp(min=1)
            x = self.multihead_self_attention(x, length=click_length)
        else:
            x = self.multihead_self_attention(x)

        x = x + residual
        x = self.layer_norm(x)

        # Dense user representation & Sparse user history representation based on aggregation variant
        if self.user_aggregation == 'attention':
            user_dense_vec = self.additive_attention(x, mask=mask)
            attn_weights = self.additive_attention(x, mask=mask, return_weights=True)
            user_sparse_hist = torch.bmm(attn_weights.unsqueeze(1), sparse_article_vectors).squeeze(1)

        elif self.user_aggregation in ['average', 'mean', 'weighted_sum']:
            if mask is not None:
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
                weights = mask / denom
                user_dense_vec = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
                user_sparse_hist = torch.bmm(weights.unsqueeze(1), sparse_article_vectors).squeeze(1)
            else:
                weights = torch.ones_like(x[:, :, 0]) / x.size(1)
                user_dense_vec = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
                user_sparse_hist = torch.bmm(weights.unsqueeze(1), sparse_article_vectors).squeeze(1)

        elif self.user_aggregation in ['max', 'max_pooling']:
            if mask is not None:
                masked_x = x.masked_fill((mask == 0).unsqueeze(-1), -1e9)
                user_dense_vec = masked_x.max(dim=1).values
                masked_sparse = sparse_article_vectors.masked_fill((mask == 0).unsqueeze(-1), 0.0)
                user_sparse_hist = masked_sparse.max(dim=1).values
            else:
                user_dense_vec = x.max(dim=1).values
                user_sparse_hist = sparse_article_vectors.max(dim=1).values

        elif self.user_aggregation in ['decay', 'temporal_decay']:
            beta = self.temporal_decay_beta
            if mask is not None:
                history_lengths = mask.sum(dim=1, keepdim=True)
                positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
                dist = (history_lengths - 1) - positions
                decay_weights = (beta ** dist) * mask
                denom = decay_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
                weights = decay_weights / denom
                user_dense_vec = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
                user_sparse_hist = torch.bmm(weights.unsqueeze(1), sparse_article_vectors).squeeze(1)
            else:
                seq_len = x.size(1)
                positions = torch.arange(seq_len, device=x.device)
                dist = (seq_len - 1) - positions
                decay_weights = beta ** dist
                weights = decay_weights / decay_weights.sum().clamp(min=1e-9)
                weights = weights.unsqueeze(0).expand(x.size(0), -1)
                user_dense_vec = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
                user_sparse_hist = torch.bmm(weights.unsqueeze(1), sparse_article_vectors).squeeze(1)

        else:
            raise ValueError(f"Unknown user aggregation variant: {self.user_aggregation}")

        # Compute projected representation from dense user representation if dense pathway is enabled
        if self.user_pathway != 'sparse':
            user_sparse_proj = self.projection(user_dense_vec)
            user_sparse_proj = torch.log1p(torch.relu(user_sparse_proj))
        else:
            user_sparse_proj = torch.zeros(user_dense_vec.shape[0], self.sparse_dim, device=user_dense_vec.device)

        # Combine exact history terms and projected semantic terms depending on pathway selection
        if self.user_pathway == 'both':
            user_vec = user_sparse_proj + user_sparse_hist
        elif self.user_pathway == 'dense':
            user_vec = user_sparse_proj
        elif self.user_pathway == 'sparse':
            user_vec = user_sparse_hist
        else:
            raise ValueError(f"Unknown user pathway: {self.user_pathway}")

        # Optional Top-K Sparsity Constraint on final user vector
        if self.topk_sparsity_queries is not None:
            k = int(self.topk_sparsity_queries)
            if 0 < k < user_vec.size(-1):
                _, indices = torch.topk(user_vec, k=k, dim=-1)
                mask_topk = torch.zeros_like(user_vec).scatter_(-1, indices, 1.0)
                user_vec = user_vec * mask_topk

        if return_proj:
            return user_vec, user_sparse_proj
        if return_dense:
            return user_vec, user_dense_vec
        return user_vec