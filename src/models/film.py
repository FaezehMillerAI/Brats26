import torch
from torch import nn


class FiLM(nn.Module):
    def __init__(self, meta_dim, num_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(meta_dim, num_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels * 2, num_channels * 2),
        )

    def forward(self, x, meta):
        if meta is None:
            return x
        params = self.fc(meta)
        gamma, beta = torch.chunk(params, 2, dim=1)
        gamma = gamma.view(gamma.size(0), -1, 1, 1, 1)
        beta = beta.view(beta.size(0), -1, 1, 1, 1)
        return x * (1 + gamma) + beta
