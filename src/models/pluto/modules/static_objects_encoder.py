import math
import torch
import torch.nn as nn

from ..layers.fourier_embedding import FourierEmbedding


class StaticObjectsEncoder(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.obj_encoder = FourierEmbedding(2, dim, 64)
        self.type_emb = nn.Embedding(4, dim)

        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.01)

    def forward(self, data):
        pos = data["static_objects"]["position"]
        heading = data["static_objects"]["heading"]
        shape = data["static_objects"]["shape"]
        category = data["static_objects"]["category"].long()
        valid_mask = data["static_objects"]["valid_mask"]  # [bs, N]

        obj_emb_tmp = self.obj_encoder(shape) + self.type_emb(category.long())
        obj_emb = torch.zeros_like(obj_emb_tmp)
        obj_emb[valid_mask] = obj_emb_tmp[valid_mask]

        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        obj_pos = torch.cat([pos, heading.unsqueeze(-1)], dim=-1)

        return obj_emb, obj_pos, ~valid_mask
