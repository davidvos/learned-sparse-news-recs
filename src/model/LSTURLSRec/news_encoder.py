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
        
        self.category_embedding = nn.Embedding(config['num_categories'],
                                          self.embedding_dim,
                                          padding_idx=0)
        
        self.multihead_self_attention = MultiHeadSelfAttention(
            self.embedding_dim, config['num_attention_heads'])
        
        self.additive_attention = AdditiveAttention(self.config['query_vector_dim'], 
                                                    self.embedding_dim)

        self.final_attention = AdditiveAttention(config['query_vector_dim'],
                                                     self.embedding_dim)
        

    def forward(self, news, type):
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

        news_vector = self.plm(**text)

        news_vector = news_vector.last_hidden_state

        news_vector = F.dropout(news_vector,
                                p=self.config['dropout_probability'],
                                training=self.training)
                        
        news_vector = self.multihead_self_attention(news_vector)

        news_vector = F.dropout(news_vector,
                                p=self.config['dropout_probability'],
                                training=self.training)
        
        category_vector = self.category_embedding(news['category'].to(self.device))
        subcategory_vector = self.category_embedding(news['subcategory'].to(self.device))

        if type == 'clicked':
            dense_news_vector = self.additive_attention(news_vector)
            dense_news_vector = torch.stack([dense_news_vector, category_vector, subcategory_vector], dim=1)
    
            dense_news_vector = self.final_attention(dense_news_vector)
            sparse_news_vector, regularization = None, None

        elif type == 'candidate':

            news_vector = torch.cat([news_vector, category_vector.unsqueeze(1), subcategory_vector.unsqueeze(1)], dim=1)
            news_vector = self.mlp(news_vector)
            
            logits = self.lm_head(news_vector)

            expansion_scores = torch.log1p(1 + torch.relu(logits))
            sparse_news_vector = torch.max(expansion_scores, dim=1).values
                            
            regularization = self.regularizer(sparse_news_vector)

            dense_news_vector = None
        else:
            dense_news_vector = self.additive_attention(news_vector)

            news_vector = torch.cat([news_vector, category_vector.unsqueeze(1), subcategory_vector.unsqueeze(1)], dim=1)
            news_vector = self.mlp(news_vector)
            logits = self.lm_head(news_vector)
            expansion_scores = torch.log1p(1 + torch.relu(logits))
            sparse_news_vector = torch.max(expansion_scores, dim=1).values

            regularization = self.regularizer(sparse_news_vector)
        
        return sparse_news_vector, dense_news_vector, regularization
