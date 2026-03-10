"""Generate trajectory predictions from a trained TrajectoryTransformer.

Uses teacher-forced greedy decoding: the full token sequence is passed through
the model in one forward pass, and the argmax at each action token position is
decoded to a continuous displacement.  This mirrors how dt_prediction_export
generates DT predictions (teacher-forced, full horizon at once).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from waymo_data_utils import ACT_SCALE, WOMDOfflineRLDataset


@dataclass
class SamplePrediction:
    sample_index: int
    scenario_index: int
    scenario_id: str
    pred_xy: np.ndarray    # (T, 2) world-frame
    true_xy: np.ndarray    # (T, 2) world-frame
    valid_mask: np.ndarray # (T,) int64
    ade_m: float
    fde_m: float


def _integrate_to_world_xy(
    actions: np.ndarray,       # (T, 2)  local-frame normalised displacements
    start_xy: np.ndarray,      # (2,)
    heading_cos: float,
    heading_sin: float,
) -> np.ndarray:
    """Integrate local-frame displacements to world-frame XY (same as DT export)."""
    T = actions.shape[0]
    world_xy = np.zeros((T, 2), dtype=np.float32)
    if T == 0:
        return world_xy
    world_xy[0] = start_xy
    x, y = float(start_xy[0]), float(start_xy[1])
    for t in range(1, T):
        xl, yl = float(actions[t, 0]), float(actions[t, 1])
        dx = xl * heading_cos - yl * heading_sin
        dy = xl * heading_sin + yl * heading_cos
        x += dx * ACT_SCALE
        y += dy * ACT_SCALE
        world_xy[t] = (x, y)
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
    beam_size: int = 1,
) -> list[SamplePrediction]:
    """
    For each trajectory in the dataset, decode predicted actions using
    teacher-forced greedy decoding (beam_size=1) or beam search (beam_size>1),
    then compute ADE/FDE against the ground-truth world-frame trajectory.
    """
    model.eval()
    predictions: list[SamplePrediction] = []
    n_samples = min(max_samples, len(dataset.trajectories))
    tps = model.tokens_per_step   # state_dim + act_dim + 1

    for i in range(n_samples):
        traj = dataset.trajectories[i]
        states: np.ndarray = traj["states"]    # (T, state_dim)
        actions: np.ndarray = traj["actions"]  # (T, act_dim)
        T = states.shape[0]

        tokens = model.tokenise(states, actions)  # (T * tps,)
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)

        pred_actions = np.zeros((T, model.act_dim), dtype=np.float32)

        if beam_size > 1:
            # Beam search one step at a time, feeding back predicted tokens.
            for t in range(T):
                ctx_end = t * tps + model.state_dim  # context ends just before action tokens
                context = token_tensor[0, :ctx_end]
                act_tokens = model.beam_search_actions(context, beam_size=beam_size, device=device)
                for d in range(model.act_dim):
                    pred_actions[t, d] = model.decode_action_token(int(act_tokens[d].item()), d)
        else:
            # Teacher-forced greedy: single forward pass, argmax at each action position.
            logits = model(token_tensor)  # (1, L, V)
            logits_np = logits[0].cpu()  # (L, V)
            for t in range(T):
                for d in range(model.act_dim):
                    # The logit at position (pos-1) predicts the token at pos.
                    pred_pos = t * tps + model.state_dim + d - 1
                    if pred_pos < 0 or pred_pos >= logits_np.shape[0]:
                        continue
                    act_start = model.action_offset + d * model.n_bins
                    local_pred = int(logits_np[pred_pos, act_start: act_start + model.n_bins].argmax().item())
                    pred_actions[t, d] = model.decode_action_token(act_start + local_pred, d)

        heading_cos = float(traj.get("heading_cos", 1.0))
        heading_sin = float(traj.get("heading_sin", 0.0))
        true_xy_raw = traj.get("future_xy")
        if true_xy_raw is None:
            anchor_xy = np.asarray(traj.get("anchor_xy", np.zeros(2, dtype=np.float32)), dtype=np.float32)
            true_xy = _integrate_to_world_xy(actions, anchor_xy, heading_cos, heading_sin)
        else:
            true_xy = np.asarray(true_xy_raw, dtype=np.float32)

        start_xy = true_xy[0]
        pred_xy = _integrate_to_world_xy(pred_actions, start_xy, heading_cos, heading_sin)

        valid_mask = np.asarray(
            traj.get("future_valid", np.ones((T,), dtype=np.int64)), dtype=np.int64
        )[:T]
        if valid_mask.shape[0] < T:
            valid_mask = np.pad(valid_mask, (0, T - valid_mask.shape[0]))

        ade_m, fde_m = _compute_ade_fde(pred_xy, true_xy[:T], valid_mask)

        scenario_id = traj.get("scenario_id", "")
        if isinstance(scenario_id, bytes):
            scenario_id = scenario_id.decode("utf-8", errors="ignore")

        predictions.append(SamplePrediction(
            sample_index=i,
            scenario_index=int(traj.get("scenario_index", -1)),
            scenario_id=str(scenario_id),
            pred_xy=pred_xy,
            true_xy=true_xy[:T],
            valid_mask=valid_mask,
            ade_m=ade_m,
            fde_m=fde_m,
        ))

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
