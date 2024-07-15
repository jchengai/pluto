from typing import Any, Callable, Dict, Optional

import torch
from torchmetrics import Metric

from .utils import sort_predictions


class minFDE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        k=6,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(minFDE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.k = k
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred, _ = sort_predictions(
                outputs["trajectory"], outputs["probability"], k=self.k
            )
            fde = torch.norm(
                pred[..., -1, :2] - target.unsqueeze(1)[..., -1, :2], p=2, dim=-1
            )
            min_fde = fde.min(-1)[0]
            self.sum += min_fde.sum()
            self.count += pred.shape[0]

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
