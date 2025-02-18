import torch
from model.NPA.news_encoder import NewsEncoder
from model.NPA.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor
from transformers import AutoModel

class NPA(torch.nn.Module):
    """
    NPA network.
    Given candidate news articles and a user's clicked news history,
    the network produces click probability scores.
    """
    def __init__(self, config, device):
        super(NPA, self).__init__()
        self.config = config
        self.device = device

        # Initialize the pretrained language model (BERT-like) using the provided model name.
        self.plm = AutoModel.from_pretrained(config['pretrained_model_name'])

        # News encoder to convert each news article into a vector.
        self.news_encoder = NewsEncoder(config, self.plm, device)
        # User encoder to aggregate clicked news representations.
        self.user_encoder = UserEncoder(config)
        # Dot-product based click predictor (shared with NAML).
        self.click_predictor = DotProductClickPredictor()

    def forward(self, candidate_news, clicked_news, clicked_news_mask):
        """
        Args:
            candidate_news: List of news dictionaries (each with keys "title", "category", "subcategory") 
                for candidate news articles. Length = 1+K.
            clicked_news: List of news dictionaries for the user's clicked news articles.
            clicked_news_mask: A list of tensors representing masks for clicked news 
                (to account for variable-length histories).
        Returns:
            click_probability: Tensor of shape (batch_size) with click probability scores.
        """
        # Encode candidate news articles.
        candidate_news_vector = torch.stack(
            [self.news_encoder(news) for news in candidate_news], dim=1
        )  # (batch_size, 1+K, embedding_dim)

        # Encode user's clicked news articles.
        clicked_news_vector = torch.stack(
            [self.news_encoder(news) for news in clicked_news], dim=1
        )  # (batch_size, num_clicked_news, embedding_dim)

        # Apply the clicked news mask.
        clicked_news_mask_tensor = torch.stack(clicked_news_mask).to(self.device)
        clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)  # (batch_size, num_clicked_news)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)               # (batch_size, num_clicked_news, 1)
        clicked_news_vector = clicked_news_vector * expanded_mask

        # Get the user vector by aggregating the clicked news vectors.
        user_vector = self.user_encoder(clicked_news_vector)  # (batch_size, embedding_dim)
        # Compute the click probability for the candidate news via a dot-product predictor.
        click_probability = self.click_predictor(candidate_news_vector, user_vector)  # (batch_size, 1+K)

        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news: A news dictionary with keys "title", "category", "subcategory"
        Returns:
            News representation: Tensor of shape (batch_size, embedding_dim)
        """
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: Tensor of shape (batch_size, num_clicked_news, embedding_dim)
        Returns:
            User representation: Tensor of shape (batch_size, embedding_dim)
        """
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: Tensor of shape (candidate_size, embedding_dim)
            user_vector: Tensor of shape (embedding_dim)
        Returns:
            Click probabilities: Tensor of shape (candidate_size)
        """
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)
        ).squeeze(dim=0) 