from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dt_metrics import EpochMetrics, MetricsAccumulator, masked_action_mse


def _forward_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = batch["states"].to(device)
    actions = batch["actions"].to(device)
    rtg = batch["returns_to_go"].to(device)
    timesteps = batch["timesteps"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pred_actions = model(states, actions, rtg, timesteps, attention_mask)
    return pred_actions, actions, attention_mask


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> EpochMetrics:
    model.train()
    accumulator = MetricsAccumulator()

    for batch in loader:
        pred_actions, target_actions, attention_mask = _forward_batch(model, batch, device)
        loss = masked_action_mse(pred_actions, target_actions, attention_mask, model.act_dim)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        accumulator.update(pred_actions.detach(), target_actions, attention_mask, loss.detach())

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
        pred_actions, target_actions, attention_mask = _forward_batch(model, batch, device)
        loss = masked_action_mse(pred_actions, target_actions, attention_mask, model.act_dim)
        accumulator.update(pred_actions, target_actions, attention_mask, loss)

    return accumulator.compute()
