import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class UserEncoder(torch.nn.Module):
    def __init__(self, config, plm):
        super(UserEncoder, self).__init__()
        self.config = config
        self.embedding_dim = plm.config.hidden_size * 3  # *3 because news vectors are concatenated (title, category, subcategory)
        
        # Replace additive attention with GRU
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            batch_first=True
        )

    def forward(self, user, clicked_news_vector):
        """
        Args:
            user: Long-term user representation (used as initial hidden state)
                shape: batch_size, embedding_dim
            clicked_news_vector: Recent clicked news representations
                shape: batch_size, num_clicked_news_a_user, embedding_dim
        Returns:
            shape: batch_size, embedding_dim
        """
        # Use long-term user vector as initial hidden state
        initial_state = user.unsqueeze(0)  # shape: 1, batch_size, embedding_dim
        
        # Process clicked news sequence with GRU
        _, final_state = self.gru(clicked_news_vector, initial_state)
        
        # Return final hidden state
        return final_state.squeeze(0)
