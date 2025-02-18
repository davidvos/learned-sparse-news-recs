import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

class L1Regularizer(nn.Module):
    def __init__(self, T: int = 5000, alpha: float = 0.02):
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

class UserEncoder(torch.nn.Module):
    def __init__(self, config, plm, lm_head, device):
        super(UserEncoder, self).__init__()
        self.config = config
        self.device = device

        self.embedding_dim = plm.config.hidden_size

        self.lm_head = lm_head

        self.multihead_self_attention = MultiHeadSelfAttention(
            self.embedding_dim, config['num_attention_heads'])
        
        self.additive_attention = AdditiveAttention(config['query_vector_dim'],
                                                    self.embedding_dim) 
        
        self.regularizer = L1Regularizer(alpha=0.02, T=5000)

    def forward(self, user_vector):
        """
        Args:
            user_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """

        user_vector = F.dropout(user_vector,
                                p=self.config['dropout_probability'],
                                training=self.training)
        
        user_vector = self.multihead_self_attention(user_vector)
        
        user_vector = F.dropout(user_vector,
                                p=self.config['dropout_probability'],
                                training=self.training)
        
        user_vector = self.additive_attention(user_vector)
        
        logits = self.lm_head(user_vector)

        expansion_scores = torch.log1p(1 + torch.relu(logits))
        final_user_vector = expansion_scores
        # final_user_vector = torch.max(expansion_scores, dim=1).values

        regularization = self.regularizer(final_user_vector)
        
        return final_user_vector, regularization
