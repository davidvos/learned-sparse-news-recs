import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

class L1Regularizer(nn.Module):
    """
    L1 regularization term that increases over time.
    This regularization term is used to encourage sparsity in news representations.
    """
    def __init__(self, T: int = 5000, alpha: float = 0.06):
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

class NewsEncoder(torch.nn.Module):
    """
    NewsEncoder class for processing news articles using a pre-trained language model (PLM).
    This class applies multi-head self-attention and additive attention mechanisms to the processed news articles.
    """
    def __init__(self, config, plm, device):
        """
        Args:
            config: configuration dictionary
            plm: pre-trained language model
            device: device to run the model on
        """
        super().__init__()
        self.plm = plm
        self.device = device

        self.embedding_dim = self.plm.config.hidden_size
        self.sparse_dim = plm.config.vocab_size

        self.multihead_self_attention = MultiHeadSelfAttention(
            self.embedding_dim, config['num_attention_heads'])
        
        self.additive_attention = AdditiveAttention(config['query_vector_dim'],
                                                    self.embedding_dim) 
        
        self.regularizer = L1Regularizer(alpha=0.0012, T=10000)

        self.projection = nn.Linear(self.embedding_dim, self.sparse_dim)
        self.projection.weight = self.plm.cls.predictions.decoder.weight

        self.num_categories = config['num_categories'] + 1

    def forward(self, news):
        """
        Forward pass through the NewsEncoder.
        Args:
            news: dictionary containing input_ids, attention_mask, category, and subcategory
        Returns:
            document_vector: processed document vector
            reg: regularization term
        """
        # Process text through PLM
        outputs = self.plm(
            input_ids=news["input_ids"].squeeze(1).to(self.device),
            attention_mask=news["attention_mask"].squeeze(1).to(self.device),
        )
        
        category_vector = torch.zeros((news['category'].shape[0], self.num_categories)).to(self.device)
        category_vector[torch.arange(news['category'].shape[0]), news['category']] = 1
        category_vector[torch.arange(news['subcategory'].shape[0]), news['subcategory']] = 1

        # Sparse activation with learned temperature
        outputs = outputs.hidden_states[-1]
        outputs = self.multihead_self_attention(outputs)
        outputs = self.additive_attention(outputs)
        logits = self.projection(outputs)
        document_vector = torch.log1p(F.relu(logits))  # [batch, seq_len, vocab_size]
        
        document_vector = torch.cat([document_vector, category_vector], dim=1)

        reg = self.regularizer(document_vector)
        return document_vector, reg