import torch
from torch import nn
from model.NRMSLSRecEarly.news_encoder import NewsEncoder
from model.NRMSLSRecEarly.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor
from transformers import AutoModel, AutoModelForMaskedLM

class NRMSLSRecEarly(torch.nn.Module):
    """
    NRMS network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, config, device):
        super(NRMSLSRecEarly, self).__init__()
        self.config = config

        self.plm = AutoModelForMaskedLM.from_pretrained(config['pretrained_model_name']).to(device)

        for name, param in self.plm.named_parameters():
            # Adjust this condition for DistilBERT (DistilBERT has fewer layers)
            if any(f"encoder.layer.{i}." in name for i in range(8)):  # Freeze layers 0-7
                param.requires_grad = False

        self.embedding_dim = self.plm.config.hidden_size
        self.vocab_size = self.plm.config.vocab_size

        self.device = device

        self.news_encoder = NewsEncoder(config, self.plm, device)
        self.user_encoder = UserEncoder(config, self.plm, device)
        self.click_predictor = DotProductClickPredictor()

    def forward(self, candidate_news, clicked_news, clicked_news_mask):
        """
        Args:
            candidate_news:
                [
                    {
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "title": batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size, 1 + K
        """
        candidate_news_tuples = [self.news_encoder(x) for x in candidate_news]
        # batch_size, 1 + K, word_embedding_dim
        candidate_news_vector = torch.stack([x[0] for x in candidate_news_tuples], dim=1)
        candidate_reg = [x[1] for x in candidate_news_tuples]

        clicked_news_tuples = [self.news_encoder(x) for x in clicked_news]
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        clicked_news_vector = torch.stack([x[0] for x in clicked_news_tuples], dim=1)
        clicked_reg = [x[1] for x in clicked_news_tuples]

        clicked_news_mask_tensor = torch.stack(clicked_news_mask).to(self.device)
        clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_news_vector = clicked_news_vector * expanded_mask

        # batch_size, word_embedding_dim
        user_vector, user_reg = self.user_encoder(clicked_news_vector)
        # batch_size, 1 + K

        click_probability = self.click_predictor(candidate_news_vector, user_vector)

        total_reg = sum(candidate_reg) + sum(clicked_reg)
        total_reg = total_reg.mean()

        return click_probability, total_reg
    
    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.news_encoder(news)[0]

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector)[0]

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
