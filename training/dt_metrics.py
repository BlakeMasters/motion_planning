from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from waymo_data_utils import ACT_SCALE


def masked_action_mse(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    attention_mask: torch.Tensor,
    action_dim: int,
) -> torch.Tensor:
    valid = attention_mask.unsqueeze(-1).float()
    return (
        (torch.nn.functional.mse_loss(pred_actions, target_actions, reduction="none") * valid).sum()
        / (valid.sum() * action_dim + 1e-8)
    )


@dataclass
class EpochMetrics:
    loss: float
    action_mse: float
    action_mae: float
    action_rmse: float
    ade_m: float
    fde_m: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class MetricsAccumulator:
    """Accumulate masked action and trajectory metrics over one epoch."""

    def __init__(self):
        self.loss_sum = 0.0
        self.loss_batches = 0
        self.action_sq_err = 0.0
        self.action_abs_err = 0.0
        self.action_count = 0.0
        self.ade_sum = 0.0
        self.ade_count = 0.0
        self.fde_sum = 0.0
        self.fde_count = 0

    def update(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        attention_mask: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        valid_step_mask = attention_mask.float()
        valid_elem_mask = valid_step_mask.unsqueeze(-1)
        err = pred_actions - target_actions

        self.loss_sum += float(loss.item())
        self.loss_batches += 1

        self.action_sq_err += float((err.pow(2) * valid_elem_mask).sum().item())
        self.action_abs_err += float((err.abs() * valid_elem_mask).sum().item())
        self.action_count += float(valid_elem_mask.sum().item() * pred_actions.shape[-1])

        pred_xy = torch.cumsum(pred_actions * ACT_SCALE, dim=1)
        true_xy = torch.cumsum(target_actions * ACT_SCALE, dim=1)
        step_dist = torch.linalg.norm(pred_xy - true_xy, dim=-1)

        self.ade_sum += float((step_dist * valid_step_mask).sum().item())
        self.ade_count += float(valid_step_mask.sum().item())

        lengths = valid_step_mask.sum(dim=1).long()
        valid_rows = lengths > 0
        if bool(valid_rows.any()):
            last_idx = (lengths - 1).clamp(min=0)
            row_idx = torch.arange(step_dist.shape[0], device=step_dist.device)
            fde = step_dist[row_idx, last_idx]
            self.fde_sum += float(fde[valid_rows].sum().item())
            self.fde_count += int(valid_rows.sum().item())

    def compute(self) -> EpochMetrics:
        action_mse = self.action_sq_err / max(self.action_count, 1e-8)
        action_mae = self.action_abs_err / max(self.action_count, 1e-8)
        action_rmse = math.sqrt(action_mse)
        ade_m = self.ade_sum / max(self.ade_count, 1e-8)
        fde_m = self.fde_sum / max(self.fde_count, 1)
        loss = self.loss_sum / max(self.loss_batches, 1)
        return EpochMetrics(
            loss=loss,
            action_mse=action_mse,
            action_mae=action_mae,
            action_rmse=action_rmse,
            ade_m=ade_m,
            fde_m=fde_m,
        )
