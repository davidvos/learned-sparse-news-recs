import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

class L1Regularizer(nn.Module):
    """
    L1 regularization term with a linearly increasing schedule for alpha.
    Encourages sparsity in representations over training steps.
    """
    def __init__(self, T: int = 5000, alpha: float = 0.06):
        """
        Args:
            T (int): Number of steps for the regularization schedule.
            alpha (float): Maximum value for the regularization coefficient.
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
            reps (Tensor): [batch_size, ...] representation tensor.
        Returns:
            reg (Tensor): Regularization loss term.
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
            # Linear schedule for alpha
            self.current_alpha = (self.current_step / self.T) * self.max_alpha
        # After T steps, alpha stays at max_alpha

class NewsEncoder(nn.Module):
    """
    Encodes news articles using a pre-trained language model (PLM),
    followed by multi-head self-attention and a projection to a sparse vocabulary space.
    """
    def __init__(self, config, plm, device):
        """
        Args:
            config (dict): Model configuration.
            plm (nn.Module): Pre-trained language model.
            device (torch.device): Device to run the model on.
        """
        super().__init__()
        self.plm = plm
        self.device = device

        self.embedding_dim = self.plm.config.hidden_size
        self.sparse_dim = plm.config.vocab_size

        # Project PLM output to vocabulary space using PLM's MLM head weights
        self.projection = nn.Linear(self.embedding_dim, self.sparse_dim)
        self.projection.weight = self.plm.cls.predictions.decoder.weight
        self.projection.bias = self.plm.cls.predictions.bias

        # Multi-head self-attention for contextualization
        self.mhsa = MultiHeadSelfAttention(self.embedding_dim, num_attention_heads=12)

        # L1 regularizer for sparsity (SPLADE-style)
        self.regularizer = L1Regularizer(alpha=0.0001, T=10000)

    def forward(self, news):
        """
        Encode news articles.
        Args:
            news (dict): Contains 'input_ids' and 'attention_mask' tensors.
        Returns:
            document_vector (Tensor): [batch, vocab_size] sparse news representation.
            reg (Tensor): Regularization loss term.
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
        document_vector = self.projection(outputs)

        # Apply log1p(ReLU(.)) for SPLADE-style sparsity
        document_vector = torch.log1p(torch.relu(document_vector))  # [batch, seq_len, vocab_size]

        # Pool over sequence (max pooling)
        document_vector = torch.max(document_vector, dim=1)[0]  # [batch, vocab_size]

        # Compute L1 regularization
        reg = self.regularizer(document_vector)

        return document_vector, reg