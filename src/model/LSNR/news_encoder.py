"""LSNR News Encoder.

Encodes raw news article text into two representations simultaneously:
a dense semantic vector (for input to the user encoder), and a sparse 
lexical expansion (for final candidate matching and user profile aggregation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import MultiHeadSelfAttention, AdditiveAttention

class NewsEncoder(nn.Module):
    """
    Encodes news articles into two representations:
    1. Sparse Vector (SPLADE-style) -> For final matching/ranking.
    2. Dense Vector -> For efficient input to the UserEncoder.
    """
    def __init__(self, config, dataset_config, plm, device):
        super().__init__()
        self.config = config
        self.plm = plm
        self.device = device
        self.embedding_dim = self.plm.config.hidden_size
        self.sparse_dim = dataset_config.get('sparse_vocab_size', plm.config.vocab_size)
        
        num_heads = config.get('num_attention_heads', 16)
        self.mhsa = MultiHeadSelfAttention(self.embedding_dim, n_head=num_heads, device=self.device)
        self.layernorm = nn.LayerNorm(self.embedding_dim)
        
        # Category embeddings (to align with checkpoint)
        self.cat_embedding = nn.Embedding(dataset_config.get('num_categories', 100), 100)
        self.subcat_embedding = nn.Embedding(dataset_config.get('num_subcategories', 1000), 100)
        self.category_projection = nn.Linear(200, self.embedding_dim)

        self.projection = nn.Linear(self.embedding_dim, self.sparse_dim, bias=False)
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

    def forward(self, news, return_sparse=True):
        inputs = self.plm(
            input_ids=news["input_ids"].squeeze(1).to(self.device),
            attention_mask=news["attention_mask"].squeeze(1).to(self.device),
        ).hidden_states[-1]

        word_mask = news["attention_mask"].squeeze(1).to(self.device)
        word_length = word_mask.sum(dim=1).to(self.device).long()
        dense_sequence = self.mhsa(inputs, length=word_length)
        dense_sequence = self.layernorm(dense_sequence + inputs)
        dense_sequence = dense_sequence * word_mask.unsqueeze(-1)

        token_counts = word_mask.sum(dim=1, keepdim=True).clamp(min=1)
        dense_vec = dense_sequence.sum(dim=1) / token_counts

        if return_sparse:
            logits = self.projection(dense_sequence)
            sparse_sequence = torch.log1p(torch.relu(logits))
            sparse_vec = torch.max(sparse_sequence, dim=1)[0]
        else:
            sparse_vec = None

        return sparse_vec, dense_vec