import torch.nn as nn

class WeightingNetwork(nn.Module):
    def __init__(self, hidden_dim=64):
        super(WeightingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Input: history length
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Output: 2 weights (for dense and sparse embeddings)
            nn.Softmax(dim=-1)         # Normalize to sum to 1
        )
    
    def forward(self, history_length):
        history_length = history_length.view(-1, 1)  # Ensure proper shape
        weights = self.net(history_length)
        return weights