from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass
class EpochMetrics:
    loss: float        # mean cross-entropy over non-padded tokens
    token_acc: float   # top-1 accuracy on non-padded tokens (fraction)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class MetricsAccumulator:
    """Accumulate CE loss and token accuracy over one epoch."""

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.loss_batches = 0
        self.correct = 0
        self.total = 0

    def update(
        self,
        logits: torch.Tensor,   # (B, L, V)
        labels: torch.Tensor,   # (B, L)  — -100 for padding
        loss: torch.Tensor,
    ) -> None:
        self.loss_sum += float(loss.item())
        self.loss_batches += 1

        valid_mask = labels != -100
        if bool(valid_mask.any()):
            pred_tokens = logits.argmax(dim=-1)          # (B, L)
            self.correct += int((pred_tokens[valid_mask] == labels[valid_mask]).sum().item())
            self.total += int(valid_mask.sum().item())

    def compute(self) -> EpochMetrics:
        loss = self.loss_sum / max(self.loss_batches, 1)
        token_acc = self.correct / max(self.total, 1)
        return EpochMetrics(loss=loss, token_acc=token_acc)
