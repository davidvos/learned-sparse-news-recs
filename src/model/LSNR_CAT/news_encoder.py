import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

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

        self.projection = nn.Linear(self.embedding_dim, self.sparse_dim)
        self.projection.weight = self.plm.cls.predictions.decoder.weight
        self.projection.bias = self.plm.cls.predictions.bias
        
        self.num_categories = config['num_categories'] + 1
        self.category_embedding = nn.Embedding(self.num_categories, self.embedding_dim, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(self.num_categories, self.embedding_dim, padding_idx=0)

        self.mhsa = MultiHeadSelfAttention(self.embedding_dim, num_attention_heads=12)


    def forward(self, news):
        """
        Forward pass through the NewsEncoder.
        Args:
            news: dictionary containing input_ids, attention_mask, category, and subcategory
        Returns:
            document_vector: processed document vector
        """
        # Process text through PLM
        outputs = self.plm(
            input_ids=news["input_ids"].squeeze(1).to(self.device),
            attention_mask=news["attention_mask"].squeeze(1).to(self.device),
        )

        # Remove one-hot encoding, use embeddings instead
        outputs = outputs.hidden_states[-1]  # [batch_size, seq_len, embedding_dim]
        category_vector = self.category_embedding(news['category'].to(self.device)).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        subcategory_vector = self.subcategory_embedding(news['subcategory'].to(self.device)).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        outputs = torch.cat([outputs, category_vector, subcategory_vector], dim=1)  # [batch_size, seq_len+2, embedding_dim]

        outputs = self.mhsa(outputs)

        # project to sparse space
        document_vector = self.projection(outputs)  # [batch_size, seq_len+2, sparse_dim]
        document_vector = torch.log1p(F.relu(document_vector))
        document_vector = torch.max(document_vector, dim=1)[0]  # [batch_size, sparse_dim]
        
        return document_vector