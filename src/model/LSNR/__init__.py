import torch
from model.LSNR.news_encoder import NewsEncoder
from model.LSNR.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor
from transformers import AutoModelForMaskedLM


class LSNR(torch.nn.Module):
    """
    NRMS network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, config, device):
        super(LSNR, self).__init__()
        self.config = config
        self.device = device
        self.plm = AutoModelForMaskedLM.from_pretrained(config['pretrained_model_name'])

        # freeze the parameters of the plm except the last 3 transformer layers and the output layers
        for layer in self.plm.bert.encoder.layer[:8]:
            for param in layer.parameters():
                param.requires_grad = False
        
        self.news_encoder = NewsEncoder(config, self.plm, device)
        self.user_encoder = UserEncoder(config, self.plm, device)

        self.click_predictor = DotProductClickPredictor()

    def forward(self, candidate_news, clicked_news, clicked_news_mask, recent_history):
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
                        "title":batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
          click_probability: batch_size, 1 + K
        """

        candidate_news_tuples = [self.news_encoder(x) for x in candidate_news]

        # batch_size, 1 + K, word_embedding_dim
        candidate_news_vector = torch.stack(
            [x[0] for x in candidate_news_tuples], dim=1)
        
        candidate_reg = [x[1] for x in candidate_news_tuples]

        user_vector, user_reg = self.user_encoder(recent_history)

        total_reg = candidate_reg
        total_reg = torch.mean(torch.stack(total_reg + [user_reg]), dim=0)

        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector,
                                                 user_vector)
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
