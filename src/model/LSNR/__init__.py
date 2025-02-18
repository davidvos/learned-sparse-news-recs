import torch
from transformers import AutoModelForMaskedLM
from model.LSNR.news_encoder import NewsEncoder
from model.LSNR.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor
import torch.nn as nn

class LSNR(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.plm = AutoModelForMaskedLM.from_pretrained(config['pretrained_model_name'], output_hidden_states=True)

        for layer in self.plm.bert.encoder.layer[:8]:
            for param in layer.parameters():
                param.requires_grad = False
        
        self.device = device

        self.news_encoder = NewsEncoder(config, self.plm, device)
        self.user_encoder = UserEncoder(config, self.plm, device)
        self.click_predictor = DotProductClickPredictor()
        
        # Add a scaling factor for regularization
        self.reg_scale = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, candidate_news, clicked_news, clicked_news_mask):
        # Compute candidate news vectors
        candidate_vecs = [self.news_encoder(x)[0] for x in candidate_news]

        # Compute clicked news vectors
        clicked_outputs = [self.news_encoder(x) for x in clicked_news]
        clicked_vecs = torch.stack([output[0] for output in clicked_outputs], dim=1)

        clicked_news_mask_tensor = torch.stack(clicked_news_mask).to(self.device)
        clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_vecs = clicked_vecs * expanded_mask

        user_vec, user_reg = self.user_encoder(clicked_vecs)
        
        # Scale regularization terms over both news and user
        news_reg = torch.mean(torch.stack([self.news_encoder.regularizer(v) for v in candidate_vecs]))
        total_reg = (user_reg + news_reg) * self.reg_scale
        
        # Compute click prediction
        click_prob = self.click_predictor(torch.stack(candidate_vecs, dim=1), user_vec)
        return click_prob, total_reg
    
    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size x num_words_title
                },
        Returns:
            (shape) batch_size x word_embedding_dim
        """
        return self.news_encoder(news)[0]

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size x num_clicked_news x word_embedding_dim
        Returns:
            (shape) batch_size x word_embedding_dim
        """
        return self.user_encoder(clicked_news_vector)[0]

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size x word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)