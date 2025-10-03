import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention


class NewsEncoder(nn.Module):
    """
    Encodes news articles using a pre-trained language model (PLM),
    followed by multi-head self-attention and a projection to a sparse vocabulary space.
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
        self.plm = plm
        self.device = device

        self.embedding_dim = self.plm.config.hidden_size
        # Use configurable sparse vocab size, fallback to PLM vocab size if not specified
        self.sparse_dim = dataset_config.get('sparse_vocab_size', plm.config.vocab_size)

        # Optional LayerNorm before projection for stability and interpretability
        self.enable_pre_projection_layernorm = bool(config.get('news_pre_projection_layernorm', False))
        if self.enable_pre_projection_layernorm:
            self.pre_projection_layernorm = nn.LayerNorm(self.embedding_dim)
        else:
            self.pre_projection_layernorm = None

        # Project PLM output to vocabulary space
        self.projection = nn.Linear(self.embedding_dim, self.sparse_dim)
        
        # Initialize weights from PLM's MLM head if dimensions match
        if self.sparse_dim == plm.config.vocab_size:
            # Use PLM's MLM head weights directly
            self.projection.weight = self.plm.cls.predictions.decoder.weight
            self.projection.bias = self.plm.cls.predictions.bias
        else:
            # For different vocab sizes (e.g., Norwegian BERT -> standard BERT vocab), 
            # initialize with Xavier uniform and copy available weights
            nn.init.xavier_uniform_(self.projection.weight)
            if self.projection.bias is not None:
                nn.init.zeros_(self.projection.bias)
            
            # Copy overlapping weights from PLM's MLM head (first sparse_dim rows)
            plm_weight = self.plm.cls.predictions.decoder.weight
            plm_bias = self.plm.cls.predictions.bias
            min_vocab_size = min(self.sparse_dim, plm.config.vocab_size)
            with torch.no_grad():
                self.projection.weight[:min_vocab_size] = plm_weight[:min_vocab_size]
                if plm_bias is not None and self.projection.bias is not None:
                    self.projection.bias[:min_vocab_size] = plm_bias[:min_vocab_size]

        # Multi-head self-attention for contextualization
        self.mhsa = MultiHeadSelfAttention(self.embedding_dim, num_attention_heads=12)

    def forward(self, news):
        """
        Encode news articles.
        Args:
            news (dict): Contains 'input_ids' and 'attention_mask' tensors.
        Returns:
            document_vector (Tensor): [batch, vocab_size] sparse news representation.
        """
        # Pass input through PLM
        outputs = self.plm(
            input_ids=news["input_ids"].squeeze(1).to(self.device),
            attention_mask=news["attention_mask"].squeeze(1).to(self.device),
        )
        # Use last hidden state
        outputs = outputs.hidden_states[-1]

        # Contextualize with multi-head self-attention
        outputs = self.mhsa(outputs)

        # Project to vocabulary space
        if self.pre_projection_layernorm is not None:
            outputs = self.pre_projection_layernorm(outputs)
        document_vector = self.projection(outputs)

        # Apply log1p(ReLU(.)) for SPLADE-style sparsity
        document_vector = torch.log1p(torch.relu(document_vector))  # [batch, seq_len, vocab_size]

        # Pool over sequence (max pooling)
        document_vector = torch.max(document_vector, dim=1)[0]  # [batch, vocab_size]

        return document_vector