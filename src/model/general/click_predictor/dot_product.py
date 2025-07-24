import torch
import torch.nn.functional as F


class DotProductClickPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductClickPredictor, self).__init__()

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, candidate_size, X (sparse SPLADE-style vectors)
            user_vector: batch_size, X (sparse SPLADE-style vectors)
        Returns:
            (shape): batch_size
        """
        # For SPLADE-style sparse vectors, magnitude matters and should NOT be normalized
        # The raw values encode term importance and sparse activation patterns
        
        # Compute dot product with raw vectors
        # batch_size, candidate_size
        probability = torch.bmm(candidate_news_vector, user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        
        # Apply temperature scaling for numerical stability
        temperature = 1.1
        probability = probability / temperature
        
        return probability
