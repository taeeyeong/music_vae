import torch
from torch import nn
import torch.nn.functional as F


class BidirectionalLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(BidirectionalLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True)

        # fully connection for Latent code
        # 2 for bidirectional
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)

        # 양방향의 마지막 상태를 연결
        h_concat = torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1)

        # Latent code
        mu = self.fc_mu(h_concat)
        sigma = torch.log(torch.exp(self.fc_sigma(h_concat)) + 1)
        
        return mu, sigma
