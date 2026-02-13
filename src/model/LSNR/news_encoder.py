import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming these imports exist in your project structure
from model.layers import MultiHeadSelfAttention, AdditiveAttention


# --- 1. News Encoder ---
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
        
        # Multi-head self-attention for contextualizing PLM outputs
        # Get head count from config (default to 16 if not found)
        num_heads = config.get('num_attention_heads', 16)
        self.mhsa = MultiHeadSelfAttention(self.embedding_dim, n_head=num_heads, device=self.device)
        
        # LayerNorm to stabilize the dense vector before branching
        self.layernorm = nn.LayerNorm(self.embedding_dim)

        # Projection from Dense to Sparse Vocabulary Space (Bias disabled to preserve sparsity)
        self.projection = nn.Linear(self.embedding_dim, self.sparse_dim, bias=False)
        
        # Initialize projection weights (smart initialization)
        self._init_projection_weights(plm)
        

        
        # Dense pooling strategy: "max" or "mean"
        self.dense_pooling = config.get('dense_pooling', 'max')



    def _init_projection_weights(self, plm):
        """Initializes projection layer using PLM's MLM head if possible."""
        if self.sparse_dim == plm.config.vocab_size:
            # Exact match: copy weights directly (Ignore bias)
            self.projection.weight = self.plm.cls.predictions.decoder.weight
        else:
            # Size mismatch: Xavier init and copy overlapping part
            nn.init.xavier_uniform_(self.projection.weight)
            
            # Copy overlapping weights (e.g., first N common tokens)
            plm_weight = self.plm.cls.predictions.decoder.weight
            min_vocab = min(self.sparse_dim, plm.config.vocab_size)
            
            with torch.no_grad():
                self.projection.weight[:min_vocab] = plm_weight[:min_vocab]

    def forward(self, news, return_sparse=True):
        """
        Args:
            news (dict): Batch of news inputs (input_ids, attention_mask, category, subcategory).
            return_sparse (bool): Whether to compute and return the sparse vector.
        Returns:
            sparse_vec (Tensor or None): [batch, vocab_size] - Ranking/Loss representation.
            dense_vec (Tensor):  [batch, hidden_size] - Efficient UserEncoder input.
        """
        # 1. Pass through Pre-trained Language Model
        inputs = self.plm(
            input_ids=news["input_ids"].squeeze(1).to(self.device),
            attention_mask=news["attention_mask"].squeeze(1).to(self.device),
        ).hidden_states[-1] # [batch, seq_len, hidden_dim]

        # 2. Contextualize with MHSA
        word_mask = news["attention_mask"].squeeze(1).to(self.device)
        word_length = word_mask.sum(dim=1).to(self.device).long()
        dense_sequence = self.mhsa(inputs, length=word_length) # [batch, seq_len, hidden_dim]
        # Residual Connection
        dense_sequence = self.layernorm(dense_sequence + inputs)
        
        # Apply mask before pooling to ensure padded tokens (which might have high BERT activations) don't contribute
        dense_sequence = dense_sequence * word_mask.unsqueeze(-1)



        # --- Branch A: Dense Representation (For User Encoder) ---
        if self.dense_pooling == 'mean':
            # Mean pooling: average over non-padded tokens
            # Mask already applied above, so sum and divide by token count
            token_counts = word_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
            dense_vec = dense_sequence.sum(dim=1) / token_counts  # [batch, hidden]
        else:
            # Max pooling (default): take max over sequence
            # Use large negative for padding to ensure max selects real values
            dense_sequence_for_pooling = dense_sequence.masked_fill(word_mask.unsqueeze(-1) == 0, -1e9)
            dense_vec = torch.max(dense_sequence_for_pooling, dim=1)[0]

        # --- Branch B: Sparse Representation (For Ranking) ---
        if return_sparse:
            # Project tokens to vocabulary space [batch, seq_len, vocab_size]
            logits = self.projection(dense_sequence)
            
            # SPLADE Activation: log(1 + ReLU(x))
            # This creates sparsity while keeping magnitudes relevant
            sparse_sequence = torch.log1p(torch.relu(logits))
            
            # Max Pool over sequence dimensions to get document-level sparsity
            sparse_vec = torch.max(sparse_sequence, dim=1)[0]
            # Note: BM25 saturation is now applied symmetrically in the click predictor
        else:
            sparse_vec = None

        return sparse_vec, dense_vec