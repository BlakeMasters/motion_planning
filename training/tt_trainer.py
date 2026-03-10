from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tt_metrics import EpochMetrics, MetricsAccumulator


def _forward_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["attention_mask"].to(device)
    logits = model(ids, attention_mask=mask)
    return logits, labels, mask


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> EpochMetrics:
    model.train()
    accumulator = MetricsAccumulator()

    for batch in loader:
        logits, labels, _ = _forward_batch(model, batch, device)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        accumulator.update(logits.detach(), labels, loss.detach())

    return accumulator.compute()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    accumulator = MetricsAccumulator()

    for batch in loader:
        logits, labels, _ = _forward_batch(model, batch, device)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        accumulator.update(logits, labels, loss)

    return accumulator.compute()
