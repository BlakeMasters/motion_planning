from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from waymo_data_utils import ACT_SCALE, WOMDOfflineRLDataset


@dataclass
class SamplePrediction:
    sample_index: int
    scenario_index: int
    scenario_id: str
    pred_xy: np.ndarray
    true_xy: np.ndarray
    valid_mask: np.ndarray
    ade_m: float
    fde_m: float


def _local_actions_to_world_xy(
    local_actions: np.ndarray,
    start_xy: np.ndarray,
    heading_cos: float,
    heading_sin: float,
) -> np.ndarray:
    """Integrate local-frame displacement actions into world-frame XY."""
    horizon = local_actions.shape[0]
    world_xy = np.zeros((horizon, 2), dtype=np.float32)
    if horizon == 0:
        return world_xy

    world_xy[0] = start_xy
    x, y = float(start_xy[0]), float(start_xy[1])

    for t in range(1, horizon):
        xl, yl = float(local_actions[t, 0]), float(local_actions[t, 1])
        dx_norm = xl * heading_cos - yl * heading_sin
        dy_norm = xl * heading_sin + yl * heading_cos
        x += dx_norm * ACT_SCALE
        y += dy_norm * ACT_SCALE
        world_xy[t, 0] = x
        world_xy[t, 1] = y

    return world_xy


def _compute_ade_fde(
    pred_xy: np.ndarray,
    true_xy: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float]:
    valid = valid_mask.astype(bool)
    if valid.sum() == 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(pred_xy - true_xy, axis=-1)
    ade = float(dist[valid].mean())
    fde = float(dist[np.where(valid)[0][-1]])
    return ade, fde


@torch.no_grad()
def generate_sample_predictions(
    model: torch.nn.Module,
    dataset: WOMDOfflineRLDataset,
    device: torch.device,
    max_samples: int,
) -> list[SamplePrediction]:
    model.eval()
    predictions: list[SamplePrediction] = []
    n_samples = min(max_samples, len(dataset.trajectories))

    for i in range(n_samples):
        traj: dict[str, Any] = dataset.trajectories[i]
        states = torch.from_numpy(traj["states"]).float().unsqueeze(0).to(device)
        actions = torch.from_numpy(traj["actions"]).float().unsqueeze(0).to(device)
        returns_to_go = torch.from_numpy(traj["rtg"][:, None]).float().unsqueeze(0).to(device)
        timesteps = torch.from_numpy(traj["timesteps"]).long().unsqueeze(0).to(device)
        attention_mask = torch.ones((1, states.shape[1]), dtype=torch.long, device=device)

        pred_actions = model(states, actions, returns_to_go, timesteps, attention_mask)[0].cpu().numpy()
        true_actions = traj["actions"]

        heading_cos = float(traj.get("heading_cos", 1.0))
        heading_sin = float(traj.get("heading_sin", 0.0))

        true_xy = traj.get("future_xy")
        if true_xy is None:
            start_xy = np.asarray(traj.get("anchor_xy", np.zeros((2,), dtype=np.float32)), dtype=np.float32)
            true_xy = _local_actions_to_world_xy(true_actions, start_xy, heading_cos, heading_sin)
        else:
            true_xy = np.asarray(true_xy, dtype=np.float32)
            start_xy = true_xy[0]

        pred_xy = _local_actions_to_world_xy(pred_actions, start_xy, heading_cos, heading_sin)
        valid_mask = np.asarray(traj.get("future_valid", np.ones((pred_xy.shape[0],), dtype=np.int64)), dtype=np.int64)
        valid_mask = valid_mask[: pred_xy.shape[0]]
        if valid_mask.shape[0] < pred_xy.shape[0]:
            valid_mask = np.pad(valid_mask, (0, pred_xy.shape[0] - valid_mask.shape[0]), constant_values=0)

        ade_m, fde_m = _compute_ade_fde(pred_xy, true_xy[: pred_xy.shape[0]], valid_mask)
        scenario_id = traj.get("scenario_id", "")
        if isinstance(scenario_id, bytes):
            scenario_id = scenario_id.decode("utf-8", errors="ignore")
        predictions.append(
            SamplePrediction(
                sample_index=i,
                scenario_index=int(traj.get("scenario_index", -1)),
                scenario_id=str(scenario_id),
                pred_xy=pred_xy,
                true_xy=true_xy[: pred_xy.shape[0]],
                valid_mask=valid_mask,
                ade_m=ade_m,
                fde_m=fde_m,
            )
        )

    return predictions


def summarise_sample_predictions(samples: list[SamplePrediction]) -> dict[str, float]:
    if not samples:
        return {"num_samples": 0}
    ade_vals = np.array([s.ade_m for s in samples], dtype=np.float32)
    fde_vals = np.array([s.fde_m for s in samples], dtype=np.float32)
    return {
        "num_samples": float(len(samples)),
        "ade_mean_m": float(np.nanmean(ade_vals)),
        "ade_median_m": float(np.nanmedian(ade_vals)),
        "fde_mean_m": float(np.nanmean(fde_vals)),
        "fde_median_m": float(np.nanmedian(fde_vals)),
    }


def save_sample_predictions(path: Path, samples: list[SamplePrediction]) -> None:
    if not samples:
        raise ValueError("No sample predictions to save.")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_index=np.array([s.sample_index for s in samples], dtype=np.int64),
        scenario_index=np.array([s.scenario_index for s in samples], dtype=np.int64),
        scenario_id=np.array([s.scenario_id for s in samples], dtype=np.str_),
        pred_xy=np.stack([s.pred_xy for s in samples], axis=0),
        true_xy=np.stack([s.true_xy for s in samples], axis=0),
        valid_mask=np.stack([s.valid_mask for s in samples], axis=0),
        ade_m=np.array([s.ade_m for s in samples], dtype=np.float32),
        fde_m=np.array([s.fde_m for s in samples], dtype=np.float32),
    )
