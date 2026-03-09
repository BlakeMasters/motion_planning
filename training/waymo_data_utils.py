"""Shared data utilities for Waymo offline RL training scripts.

Used by both train_decision_transformer_gcs.py and
train_trajectory_transformer_gcs.py.
"""

from __future__ import annotations  # allows X | Y unions on Python 3.9

import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# tensorflow and google-cloud-storage are only needed for the TFRecord data
# pipeline (build_features / parse_scenario / build_tf_dataset /
# check_gcs_access).  They are imported lazily inside those functions so the
# rest of the module — models, datasets, smoke tests — can load without them.


# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------

POS_SCALE = 50.0
VEL_SCALE = 15.0
ACT_SCALE = 1.5
K_NEAREST = 5


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def gcs_bucket_from_path(path: str) -> str | None:
    if not path.startswith("gs://"):
        return None
    return path[5:].split("/", 1)[0]


def check_gcs_access(bucket: str) -> bool:
    try:
        from google.cloud import storage
        project = (
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT")
        )
        client = storage.Client(project=project) if project else storage.Client()
        b = client.bucket(bucket)
        next(client.list_blobs(b, max_results=1))
        return True
    except Exception as exc:
        print(f"GCS not accessible: {exc}")
        return False


def validate_gcs_access(train_path: str, val_path: str) -> bool:
    buckets = sorted(
        {
            b
            for b in (
                gcs_bucket_from_path(train_path),
                gcs_bucket_from_path(val_path),
            )
            if b is not None
        }
    )
    if not buckets:
        return True
    return all(check_gcs_access(bucket) for bucket in buckets)


# ---------------------------------------------------------------------------
# TFRecord / dataset construction
# ---------------------------------------------------------------------------

def build_features(n_agents: int, n_past: int, n_current: int, n_future: int) -> dict:
    import tensorflow as tf
    state_features: dict = {}
    for split, steps in [("past", n_past), ("current", n_current), ("future", n_future)]:
        for feat in ["x", "y", "bbox_yaw", "velocity_x", "velocity_y", "valid"]:
            dtype = tf.int64 if feat == "valid" else tf.float32
            state_features[f"state/{split}/{feat}"] = tf.io.FixedLenFeature(
                [n_agents, steps], dtype
            )
    for key in ("is_sdc", "type"):
        dtype = tf.int64 if key == "is_sdc" else tf.float32
        state_features[f"state/{key}"] = tf.io.FixedLenFeature([n_agents], dtype)
    return state_features


def list_shards(path: str, n: int) -> list[str]:
    if path.startswith("gs://"):
        from google.cloud import storage
        parts = path[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        client = storage.Client()
        blobs = list(client.list_blobs(bucket, prefix=prefix, max_results=n * 4))
        paths = [
            f"gs://{bucket}/{blob.name}"
            for blob in blobs
            if blob.name.endswith(".tfrecord") or "tfrecord" in blob.name
        ]
        paths.sort()
        return paths[:n]

    from glob import glob
    paths = sorted(glob(os.path.join(path, "*.tfrecord*")))
    return paths[:n]


def parse_scenario(raw: bytes, features: dict) -> dict:
    import tensorflow as tf
    ex = tf.io.parse_single_example(raw, features)

    def _stack_fields(split: str):
        return tf.stack(
            [
                ex[f"state/{split}/x"],
                ex[f"state/{split}/y"],
                ex[f"state/{split}/velocity_x"],
                ex[f"state/{split}/velocity_y"],
                tf.math.cos(ex[f"state/{split}/bbox_yaw"]),
                tf.math.sin(ex[f"state/{split}/bbox_yaw"]),
            ],
            axis=-1,
        )

    past = _stack_fields("past")
    cur = _stack_fields("current")
    future = _stack_fields("future")
    fut_valid = tf.concat(
        [ex["state/past/valid"], ex["state/current/valid"], ex["state/future/valid"]],
        axis=1,
    )
    return {
        "history": tf.concat([past, cur], axis=1),
        "future": future,
        "fut_valid": fut_valid,
        "is_sdc": ex["state/is_sdc"],
        "type": ex["state/type"],
    }


def build_tf_dataset(
    path: str,
    n_shards: int,
    n_agents: int,
    n_past: int,
    n_current: int,
    n_future: int,
) -> Optional[object]:
    import tensorflow as tf
    shards = list_shards(path, n_shards)
    if not shards:
        return None
    print(f"Using {len(shards)} shards from {path}")
    print(f"First shard: {shards[0]}")
    features = build_features(n_agents, n_past, n_current, n_future)
    ds = (
        tf.data.TFRecordDataset(shards, num_parallel_reads=4)
        .map(lambda raw: parse_scenario(raw, features), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(1)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


# ---------------------------------------------------------------------------
# State / action / reward construction
# ---------------------------------------------------------------------------

def build_state(hist: np.ndarray, sdc_idx: int) -> np.ndarray:
    """Build a 16-dim ego-centric state vector from the last history frame."""
    t = hist.shape[1] - 1
    ex, ey, evx, evy, cosh, sinh = hist[sdc_idx, t]

    state = np.zeros((16,), dtype=np.float32)
    state[0] = ex / POS_SCALE
    state[1] = ey / POS_SCALE
    state[2] = cosh
    state[3] = sinh
    state[4] = evx / VEL_SCALE
    state[5] = evy / VEL_SCALE

    others = []
    for i in range(hist.shape[0]):
        if i == sdc_idx:
            continue
        ox, oy = hist[i, t, 0], hist[i, t, 1]
        if ox == 0 and oy == 0:
            continue
        dist = math.sqrt((ox - ex) ** 2 + (oy - ey) ** 2)
        others.append((dist, i))
    others.sort()

    slot = 6
    for _, idx in others[:K_NEAREST]:
        state[slot] = (hist[idx, t, 0] - ex) / POS_SCALE
        state[slot + 1] = (hist[idx, t, 1] - ey) / POS_SCALE
        slot += 2
    return state


def build_action(fut: np.ndarray, hist: np.ndarray, sdc_idx: int, t: int) -> np.ndarray:
    """Build a 2-dim local-frame displacement action for timestep t."""
    if t == 0:
        return np.zeros((2,), dtype=np.float32)
    dx = (fut[sdc_idx, t, 0] - fut[sdc_idx, t - 1, 0]) / ACT_SCALE
    dy = (fut[sdc_idx, t, 1] - fut[sdc_idx, t - 1, 1]) / ACT_SCALE
    cos_h = hist[sdc_idx, -1, 4]
    sin_h = hist[sdc_idx, -1, 5]
    xl = dx * cos_h + dy * sin_h
    yl = -dx * sin_h + dy * cos_h
    return np.array([xl, yl], dtype=np.float32)


def compute_rtg(
    fut: np.ndarray, sdc_idx: int, horizon: int, rtg_scale: float
) -> np.ndarray:
    """Compute clipped return-to-go over the future horizon."""
    rewards = np.zeros((horizon,), dtype=np.float32)
    for t in range(1, horizon):
        dx = fut[sdc_idx, t, 0] - fut[sdc_idx, t - 1, 0]
        dy = fut[sdc_idx, t, 1] - fut[sdc_idx, t - 1, 1]
        rewards[t] = -math.sqrt(dx**2 + dy**2)

    rtg = np.zeros((horizon,), dtype=np.float32)
    running = 0.0
    for t in range(horizon - 1, -1, -1):
        running += rewards[t]
        rtg[t] = running / rtg_scale
    return np.clip(rtg, -1.0, 0.0)


# ---------------------------------------------------------------------------
# Shared PyTorch Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Subset of TrainConfig fields consumed by WOMDOfflineRLDataset."""
    state_dim: int
    act_dim: int
    context_len: int
    pred_horizon: int
    rtg_scale: float


class WOMDOfflineRLDataset(Dataset):
    """
    Offline RL dataset built from WOMD TFRecords.

    When ``tf_dataset`` is None, falls back to synthetic circular
    trajectories so the architecture can be exercised without GCS
    credentials or TensorFlow installed.

    Each item is a dict with keys:
        states          : (context_len, state_dim)   float32
        actions         : (context_len, act_dim)     float32
        returns_to_go   : (context_len, 1)           float32
        timesteps       : (context_len,)             int64
        attention_mask  : (context_len,)             int64   1=real 0=pad
    """

    def __init__(
        self,
        tf_dataset: Optional[object],
        max_scenarios: int,
        cfg: DatasetConfig,
    ):
        self.context_len = cfg.context_len
        self.pred_horizon = cfg.pred_horizon
        self.state_dim = cfg.state_dim
        self.act_dim = cfg.act_dim
        self.rtg_scale = cfg.rtg_scale
        self.windows_per_traj = max(1, cfg.pred_horizon - cfg.context_len + 1)
        self.trajectories: list[dict[str, np.ndarray]] = []

        if tf_dataset is None:
            self._make_synthetic(max_scenarios)
        else:
            self._load_from_tf(tf_dataset, max_scenarios, cfg)

    def _load_from_tf(self, tf_dataset: object, max_scenarios: int, cfg: DatasetConfig) -> None:
        for i, sc in enumerate(tf_dataset):
            if i >= max_scenarios:
                break
            hist = sc["history"][0].numpy()
            fut = sc["future"][0].numpy()
            sdc = sc["is_sdc"][0].numpy()

            sdc_idxs = np.where(sdc > 0)[0]
            if len(sdc_idxs) == 0:
                continue
            sdc_idx = int(sdc_idxs[0])

            T = cfg.pred_horizon
            states = np.array(
                [build_state(hist, sdc_idx) for _ in range(T)], dtype=np.float32
            )
            actions = np.array(
                [build_action(fut, hist, sdc_idx, t) for t in range(T)], dtype=np.float32
            )
            rtg = compute_rtg(fut, sdc_idx, T, cfg.rtg_scale)
            self.trajectories.append(
                {
                    "states": states,
                    "actions": actions,
                    "rtg": rtg,
                    "timesteps": np.arange(T, dtype=np.int64),
                }
            )

        if not self.trajectories:
            raise RuntimeError("No trajectories extracted. Check shards/auth/config.")

    def _make_synthetic(self, n: int) -> None:
        """Synthetic circular trajectories for smoke-testing without real data."""
        print(f"No tf_dataset provided — generating {n} synthetic trajectories.")
        T = self.pred_horizon
        for _ in range(n):
            radius = np.random.uniform(5, 30)
            omega = np.random.uniform(0.05, 0.2)
            phase = np.random.uniform(0, 2 * math.pi)

            states = np.zeros((T, self.state_dim), dtype=np.float32)
            actions = np.zeros((T, self.act_dim), dtype=np.float32)
            rtg = np.zeros(T, dtype=np.float32)

            for t in range(T):
                angle = phase + omega * t
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                dx = -radius * omega * math.sin(angle) * 0.1
                dy = radius * omega * math.cos(angle) * 0.1
                states[t, 0] = x / POS_SCALE
                states[t, 1] = y / POS_SCALE
                actions[t] = np.array([dx, dy]) / ACT_SCALE
                step_len = math.sqrt(dx**2 + dy**2)
                rtg[t] = max(-1.0, -(T - t) * step_len / self.rtg_scale)

            self.trajectories.append(
                {
                    "states": states,
                    "actions": actions,
                    "rtg": rtg,
                    "timesteps": np.arange(T, dtype=np.int64),
                }
            )

    def __len__(self) -> int:
        return len(self.trajectories) * self.windows_per_traj

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_idx = idx // self.windows_per_traj
        start = idx % self.windows_per_traj
        traj = self.trajectories[traj_idx]
        T = self.pred_horizon
        CL = self.context_len

        def window(arr: np.ndarray) -> np.ndarray:
            end = start + CL
            if end <= T:
                return arr[start:end].copy()
            pad = end - T
            return np.concatenate(
                [arr[start:T], np.zeros((pad,) + arr.shape[1:], dtype=arr.dtype)], axis=0
            )

        states = window(traj["states"])
        actions = window(traj["actions"])
        rtg = window(traj["rtg"][:, None])
        timesteps = traj["timesteps"][start : start + CL]
        if len(timesteps) < CL:
            timesteps = np.concatenate(
                [timesteps, np.full((CL - len(timesteps),), timesteps[-1] + 1, dtype=np.int64)]
            )

        real = min(CL, T - start)
        mask = np.array([1] * real + [0] * (CL - real), dtype=np.int64)

        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "returns_to_go": torch.tensor(rtg, dtype=torch.float32),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# TT-specific dataset and collate (used by train_trajectory_transformer_gcs.py)
# ---------------------------------------------------------------------------

class _Tokeniser(Protocol):
    """Structural interface expected from TrajectoryTransformer.tokenise."""
    def tokenise(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray: ...


class TTTokenDataset(Dataset):
    """
    Wraps WOMDOfflineRLDataset and converts each window into a flat token
    sequence for the Trajectory Transformer.

    Token sequence per timestep  (length = state_dim + act_dim + 1):
        s[0], ..., s[state_dim-1],  a[0], ..., a[act_dim-1],  r_placeholder

    The reward placeholder is always the same token (structural separator).
    Labels are the input sequence shifted left by 1 (standard LM training).
    Padded positions use label = -100 so cross-entropy ignores them.
    """

    def __init__(self, rl_dataset: WOMDOfflineRLDataset, model: _Tokeniser):
        self.rl_dataset = rl_dataset
        self.model = model

    def __len__(self) -> int:
        return len(self.rl_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.rl_dataset[idx]
        states = sample["states"].numpy()        # (CL, state_dim)
        actions = sample["actions"].numpy()      # (CL, act_dim)
        mask = sample["attention_mask"].numpy()  # (CL,)

        T = int(mask.sum())
        tokens = self.model.tokenise(states[:T], actions[:T])  # (T * tps,)

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def tt_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length token sequences to equal length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for b in batch:
        L = b["input_ids"].shape[0]
        pad = max_len - L
        input_ids_list.append(F.pad(b["input_ids"], (0, pad), value=0))
        labels_list.append(F.pad(b["labels"], (0, pad), value=-100))
        mask_list.append(
            torch.cat([torch.ones(L, dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
        )
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }
