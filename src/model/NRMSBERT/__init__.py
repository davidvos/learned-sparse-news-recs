import torch
from model.NRMSBERT.news_encoder import NewsEncoder
from model.NRMSBERT.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor
from transformers import AutoModel

class NRMSBERT(torch.nn.Module):
    """
    NRMS network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self, config, device):
        super(NRMSBERT, self).__init__()
        self.config = config

        self.plm = AutoModel.from_pretrained(config['pretrained_model_name'])

        for layer in self.plm.encoder.layer[:8]:
            for param in layer.parameters():
                param.requires_grad = False
                
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
                        "title":batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
          click_probability: batch_size, 1 + K
        """
        # Batch process candidate news
        num_candidates = len(candidate_news)
        batch_size = candidate_news[0]['input_ids'].shape[0]
        
        # Concatenate all candidate news into a single batch
        batched_candidate = {
            'input_ids': torch.cat([news['input_ids'] for news in candidate_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in candidate_news], dim=0),
        }

        print(batched_candidate['input_ids'].shape)
        
        # Process all candidate news in a single forward pass
        candidate_vecs_batched = self.news_encoder(batched_candidate)
        
        # Reshape results back to original structure: batch_size, num_candidates, embedding_dim
        candidate_news_vector = candidate_vecs_batched.view(num_candidates, batch_size, -1).transpose(0, 1)
        
        # Batch process clicked news
        num_clicked = len(clicked_news)
        
        # Concatenate all clicked news into a single batch
        batched_clicked = {
            'input_ids': torch.cat([news['input_ids'] for news in clicked_news], dim=0),
            'attention_mask': torch.cat([news['attention_mask'] for news in clicked_news], dim=0),
        }
        
        # Process all clicked news in a single forward pass
        clicked_vecs_batched = self.news_encoder(batched_clicked)
        
        # Reshape results: batch_size, num_clicked, embedding_dim
        clicked_news_vector = clicked_vecs_batched.view(num_clicked, batch_size, -1).transpose(0, 1)
        
        # Apply mask to clicked news vectors
        clicked_news_mask_tensor = torch.stack(clicked_news_mask).to(self.device)
        clicked_news_mask_tensor = clicked_news_mask_tensor.transpose(0, 1)
        expanded_mask = clicked_news_mask_tensor.unsqueeze(-1)
        clicked_news_vector = clicked_news_vector * expanded_mask
        
        # Get user vector
        user_vector = self.user_encoder(clicked_news_vector)
        
        # Compute click probability
        click_probability = self.click_predictor(candidate_news_vector, user_vector)
        
        return click_probability

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
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector)

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
