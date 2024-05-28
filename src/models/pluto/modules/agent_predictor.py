import torch
import torch.nn as nn

from ..layers.mlp_layer import MLPLayer


class AgentPredictor(nn.Module):
    def __init__(self, dim, future_steps) -> None:
        super().__init__()

        self.future_steps = future_steps

        self.loc_predictor = MLPLayer(dim, 2 * dim, future_steps * 2)
        self.yaw_predictor = MLPLayer(dim, 2 * dim, future_steps * 2)
        self.vel_predictor = MLPLayer(dim, 2 * dim, future_steps * 2)

    def forward(self, x):
        """
        x: (bs, N, dim)
        """

        bs, N, _ = x.shape

        loc = self.loc_predictor(x).view(bs, N, self.future_steps, 2)
        yaw = self.yaw_predictor(x).view(bs, N, self.future_steps, 2)
        vel = self.vel_predictor(x).view(bs, N, self.future_steps, 2)

        prediction = torch.cat([loc, yaw, vel], dim=-1)
        return prediction
