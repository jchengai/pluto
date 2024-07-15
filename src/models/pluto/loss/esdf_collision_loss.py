import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ESDFCollisionLoss(nn.Module):
    def __init__(
        self,
        num_circles=3,
        ego_width=2.297,
        ego_front_length=4.049,
        ego_rear_length=1.127,
        resolution=0.2,
    ) -> None:
        super().__init__()

        ego_length = ego_front_length + ego_rear_length
        interval = ego_length / num_circles

        self.N = num_circles
        self.width = ego_width
        self.length = ego_length
        self.rear_length = ego_rear_length
        self.resolution = resolution

        self.radius = math.sqrt(ego_width**2 + interval**2) / 2 - resolution
        self.offset = torch.Tensor(
            [-ego_rear_length + interval / 2 * (2 * i + 1) for i in range(num_circles)]
        )

    def forward(self, trajectory: Tensor, sdf: Tensor):
        """
        trajectory: (bs, T, 4) - [x, y, cos0, sin0]
        sdf: (bs, H, W)
        """
        bs, H, W = sdf.shape

        origin_offset = torch.tensor([W // 2, H // 2], device=sdf.device)
        offset = self.offset.to(sdf.device).view(1, 1, self.N, 1)
        # (bs, T, N, 2)
        centers = trajectory[..., None, :2] + offset * trajectory[..., None, 2:4]

        pixel_coord = torch.stack(
            [centers[..., 0] / self.resolution, -centers[..., 1] / self.resolution],
            dim=-1,
        )
        grid_xy = pixel_coord / origin_offset
        valid_mask = (grid_xy < 0.95).all(-1) & (grid_xy > -0.95).all(-1)
        on_road_mask = sdf[:, H // 2, W // 2] > 0

        # (bs, T, N)
        distance = F.grid_sample(
            sdf.unsqueeze(1), grid_xy, mode="bilinear", padding_mode="zeros"
        ).squeeze(1)

        cost = self.radius - distance
        valid_mask = valid_mask & (cost > 0) & on_road_mask[:, None, None]
        cost.masked_fill_(~valid_mask, 0)

        loss = F.l1_loss(cost, torch.zeros_like(cost), reduction="none").sum(-1)
        loss = loss.sum() / (valid_mask.sum() + 1e-6)

        return loss
