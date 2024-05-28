import torch.nn as nn


class MLPLayer(nn.Module):
    def __init__(self, channel_in, hidden, channel_out) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(channel_in, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel_out),
        )

    def forward(self, x):
        return self.mlp(x)
