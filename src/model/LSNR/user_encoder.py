import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming these imports exist in your project structure
from model.layers import MultiHeadSelfAttention, AdditiveAttention

# --- 2. User Encoder ---
class UserEncoder(nn.Module):
    """
    Encodes user history using DENSE news vectors (efficient) and outputs a SPARSE user vector.
    Removed positional embeddings for fair comparison with Bag-of-News baselines.
    """
    def __init__(self, config, dataset_config, plm, device, tied_projection_layer=None):
        super().__init__()
        self.device = device
        self.hidden_size = plm.config.hidden_size
        # Configurable sparse output dimension
        self.sparse_dim = dataset_config.get('sparse_vocab_size', plm.config.vocab_size)
        self.dropout_prob = config.get('dropout_probability', 0.1)
        self.max_history_len = config.get('max_history_len', 50)
        
        # Sparse Aggregation Configs
        self.use_sparse_aggregation = config.get('use_sparse_aggregation', False)
        self.sparse_aggregation_type = config.get('sparse_aggregation_type', 'attention_sum')
        
        # BM25 Saturation Config (for fallback dense path)
        self.use_bm25_saturation = config.get('use_bm25_saturation', False)
        self.bm25_k1 = config.get('bm25_k1', 1.2)

        # 1. Positional Embeddings REMOVED
        # self.pos_embedding = nn.Parameter(torch.randn(1, self.max_history_len, self.hidden_size))

        # 2. Temporal Modeling (Self-Attention)
        num_heads = config.get('num_attention_heads', 16)
        self.multihead_self_attention = MultiHeadSelfAttention(self.hidden_size, n_head=num_heads, device=self.device)
        
        # --- Standard Dense Aggregation (Legacy fallback) ---
        self.additive_attention = AdditiveAttention(
            query_vector_dim=config.get('query_vector_dim', 200), 
            candidate_vector_dim=self.hidden_size
        )
        
        # --- Explicit Layers for Sparse Aggregation (Masked) ---
        self.attn_W = nn.Linear(self.hidden_size, config.get('query_vector_dim', 200))
        self.attn_v = nn.Linear(config.get('query_vector_dim', 200), 1, bias=False)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Projection layer for dense aggregation path
        # Asymmetric projection: When enabled, UserEncoder uses its own projection
        # instead of sharing weights with NewsEncoder. This allows different
        # vocabulary distributions for user intent vs news content.
        self.use_asymmetric_projection = config.get('use_asymmetric_projection', False)
        
        if self.use_asymmetric_projection:
            # Create independent projection for user queries
            # Initialize from BERT's MLM head for strong starting point, then allow divergence
            self.projection = nn.Linear(self.hidden_size, self.sparse_dim, bias=False)
            self._init_projection_from_mlm(plm)
        elif tied_projection_layer is not None:
            # Tied weights with NewsEncoder (default behavior)
            self.projection = tied_projection_layer
        else:
            # Fallback: create own projection if no tied layer provided
            self.projection = nn.Linear(self.hidden_size, self.sparse_dim, bias=False)
    
    def _init_projection_from_mlm(self, plm):
        """Initialize projection weights from PLM's MLM head for better starting point."""
        if self.sparse_dim == plm.config.vocab_size:
            # Exact match: copy weights directly
            with torch.no_grad():
                self.projection.weight.copy_(plm.cls.predictions.decoder.weight)
        else:
            # Size mismatch: Xavier init + copy overlapping part
            nn.init.xavier_uniform_(self.projection.weight)
            min_vocab = min(self.sparse_dim, plm.config.vocab_size)
            with torch.no_grad():
                self.projection.weight[:min_vocab].copy_(plm.cls.predictions.decoder.weight[:min_vocab])

    def forward(self, dense_article_vectors, sparse_article_vectors=None, mask=None):
        """
        Args:
            dense_article_vectors: [batch, seq, hidden]
            sparse_article_vectors: [batch, seq, vocab]
            mask: [batch, seq] (1=Real, 0=Padding)
        """
        seq_len = dense_article_vectors.size(1)
        
        # 1. Positional Embeddings REMOVED
        # pos_emb = self.pos_embedding[:, :seq_len, :]
        # x = dense_article_vectors + pos_emb
        x = dense_article_vectors
        
        # 2. Temporal Interaction (Self-Attention)
        residual = x
        if mask is not None:
            click_length = mask.sum(dim=1).to(self.device).long()
            # Handle case where all lengths are 0 (empty history) - though usually filtered out
            # clamp to min 1 to avoid cuda errors in some implementations if 0
            click_length = click_length.clamp(min=1) 
            x = self.multihead_self_attention(x, length=click_length)
        else:
            x = self.multihead_self_attention(x)
        
        # --- Residual Connection ---
        # Maintains token identity and stabilizes gradient flow
        x = x + residual

        # --- Stabilization (Pre-Aggregation LayerNorm) ---
        x = self.layer_norm(x)

        # 3. Aggregation
        if self.use_sparse_aggregation and sparse_article_vectors is not None:
            if self.sparse_aggregation_type == 'attention_sum':
                # --- Manual Masked Attention ---
                # Project: [batch, seq, hidden] -> [batch, seq, query_dim]
                # Apply dropout to input of attention mechanism
                projected = torch.tanh(self.attn_W(self.dropout(x)))
                # Score: [batch, seq, 1]
                scores = self.attn_v(projected)
                
                if mask is not None:
                    # Set scores of padded items to -inf
                    scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
                
                # Softmax: [batch, seq, 1]
                attn_weights = F.softmax(scores, dim=1)
                
                # Weighted sum: [batch, 1, seq] @ [batch, seq, vocab] -> [batch, 1, vocab]
                # sparse_article_vectors = self.dropout(sparse_article_vectors) # DISABLED per user request
                user_emb_sparse = torch.bmm(attn_weights.transpose(1, 2), sparse_article_vectors).squeeze(1)
                
            elif self.sparse_aggregation_type == 'max':
                # For Max Pooling, zero out padded items first
                if mask is not None:
                    sparse_article_vectors = sparse_article_vectors * mask.unsqueeze(-1)
                
                # sparse_article_vectors = self.dropout(sparse_article_vectors) # DISABLED per user request
                user_emb_sparse = torch.max(sparse_article_vectors, dim=1)[0]
            else:
                raise ValueError(f"Unknown sparse_aggregation_type: {self.sparse_aggregation_type}")
        else:
            # Traditional Dense Aggregation (AdditiveAttention handles its own logic, now supports masking)
            user_emb_dense = self.additive_attention(x, mask=mask) 
            user_emb_dense = self.layer_norm(user_emb_dense)
            user_emb_proj = self.projection(user_emb_dense)
            user_emb_sparse = torch.log1p(torch.relu(user_emb_proj))
            # Note: BM25 saturation is now applied symmetrically in the click predictor

        return user_emb_sparse