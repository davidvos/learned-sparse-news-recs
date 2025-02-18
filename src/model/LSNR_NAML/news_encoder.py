import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.additive import AdditiveAttention
from model.general.attention.multihead_self import MultiHeadSelfAttention
from transformers import AutoModel

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
    def __init__(self, config, plm, mlp, lm_head, device):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.device = device

        self.plm = plm
        self.mlp = mlp
        self.lm_head = lm_head

        self.embedding_dim = self.plm.config.hidden_size
        self.regularizer = L1Regularizer(alpha=0.008, T=5000)

        self.num_categories = config['num_categories'] + 1
        

    def forward(self, news):
        """
        Args:
            news:
                {
                    "category": batch_size,
                    "subcategory": batch_size,
                    "title": batch_size * num_words_title,
                    "abstract": batch_size * num_words_abstract,
                }
        Returns:
            (shape) batch_size, num_filters
        """
        text = {
            "input_ids": news["input_ids"].squeeze(1).to(self.device),
            "attention_mask": news["attention_mask"].squeeze(1).to(self.device)
        }

        category_vector = torch.zeros((news['category'].shape[0], self.num_categories)).to(self.device)
        category_vector[torch.arange(news['category'].shape[0]), news['category']] = 1
        category_vector[torch.arange(news['subcategory'].shape[0]), news['subcategory']] = 1

        outputs = self.plm(**text)

        expansion_scores = torch.log(1 + torch.relu(outputs.logits))

        document_vector = torch.max(expansion_scores, dim=1).values

        # append category vector to the document vector
        document_vector = torch.cat([document_vector, category_vector], dim=1)

        regularization = self.regularizer(document_vector)
        
        return document_vector, regularization
