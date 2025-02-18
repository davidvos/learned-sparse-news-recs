import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

class L1Regularizer(nn.Module):
    def __init__(self, T: int = 5000, alpha: float = 0.06):
        """
        Parameters
        ---------
        T: int 
            number of warming up steps
        alpha: float 
            regularization weight
        """
        super().__init__()
        self.T = T
        self.max_alpha = alpha
        self.current_step = 0
        self.current_alpha = 0

    def forward(self, reps):
        """
        Calculate L1 for an input tensor and perform a warming up step for the regularization weight
        Parameters
        ----------
        reps: torch.Tensor
            Two dimensional input Tensor
        Returns
        -------
        torch.Tensor:
            The result of L1 applied in the input tensor. 
            L1(reps) = current_alpha * mean(L1(reps_i))
        """
        reg = torch.abs(reps).sum(dim=1).mean() * self.current_alpha
        self.step()
        return reg

    def step(self):
        """
        Perform a warming up step. This warming up step would allow us to apply the regularization gradually,
        step-by-step with increasing weight over time. Without this warming up, the loss might be overwhelmed by
        the regularization right from the start, leading to the issue that the model produces very sparse but
        not semantically useful vectors in the end.
        Parameters
        ----------
        This method does not have any input parameters
        Returns
        -------
        This method does not return anything
        """
        if self.current_step < self.T:
            self.current_step += 1
            self.current_alpha = (self.current_step / self.T) ** 2 * self.max_alpha
        else:
            pass

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, plm, device):
        super().__init__()
        self.plm = plm
        self.device = device

        self.embedding_dim = self.plm.config.hidden_size
        self.sparse_dim = plm.config.vocab_size

        self.multihead_self_attention = MultiHeadSelfAttention(
            self.embedding_dim, config['num_attention_heads'])
        
        self.additive_attention = AdditiveAttention(config['query_vector_dim'],
                                                    self.embedding_dim) 
        
        self.projection = nn.Linear(self.embedding_dim, self.sparse_dim)
        self.projection.weight = self.plm.cls.predictions.decoder.weight
        
        self.regularizer = L1Regularizer(alpha=0.0012, T=3000)
        

    def forward(self, news):

        # Process text through PLM
        outputs = self.plm(
            input_ids=news["input_ids"].squeeze(1).to(self.device),
            attention_mask=news["attention_mask"].squeeze(1).to(self.device),
        )
        
        outputs = outputs.hidden_states[-1]

        # Apply multihead self-attention
        outputs = self.multihead_self_attention(outputs)

        # Apply additive attention
        outputs = self.additive_attention(outputs)
        
        # projection is mlm head
        logits = self.projection(outputs)

        document_vector = torch.log1p(F.relu(logits))  # [batch, seq_len, vocab_size]

        reg = self.regularizer(document_vector)
        return document_vector, reg