"""Shared data utilities for Waymo offline RL training scripts.

Used by both train_decision_transformer_gcs.py and
train_trajectory_transformer_gcs.py.
"""

from __future__ import annotations  # allows X | Y unions on Python 3.9

import math
import os
import random
import time
from dataclasses import dataclass, field
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
# Artifact helpers (visualizations + GCS upload)
# ---------------------------------------------------------------------------

def upload_to_gcs(local_path: str, gcs_dir: str) -> None:
    """Upload a local file to a GCS directory (gs://bucket/prefix/).

    Called after saving checkpoints / plots so they land in AIP_MODEL_DIR.
    """
    from google.cloud import storage as gcs_storage

    if not gcs_dir.startswith("gs://"):
        return
    parts = gcs_dir[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
    blob_name = f"{prefix}/{os.path.basename(local_path)}" if prefix else os.path.basename(local_path)

    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} → gs://{bucket_name}/{blob_name}")


def save_training_plot(
    train_losses: list[float],
    val_losses: list[float],
    metric_label: str,
    local_path: str,
) -> None:
    """Save a training-curve PNG. Uploads to AIP_MODEL_DIR if set."""
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend safe for headless training
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label=f"Train {metric_label}", color="#1f77b4")
    ax.plot(epochs, val_losses,   label=f"Val {metric_label}",   color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(local_path, dpi=120)
    plt.close(fig)
    print(f"Saved training curves → {local_path}")

    aip_model_dir = os.environ.get("AIP_MODEL_DIR", "")
    if aip_model_dir:
        try:
            upload_to_gcs(local_path, aip_model_dir)
        except Exception as exc:
            print(f"Warning: could not upload plot to GCS: {exc}")


def upload_checkpoint(local_path: str) -> None:
    """Upload a checkpoint file to AIP_MODEL_DIR if the env var is set."""
    aip_model_dir = os.environ.get("AIP_MODEL_DIR", "")
    if aip_model_dir:
        try:
            upload_to_gcs(local_path, aip_model_dir)
        except Exception as exc:
            print(f"Warning: could not upload checkpoint to GCS: {exc}")


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


def validate_gcs_access(train_path: str, val_path: str, test_path: str | None = None) -> bool:
    buckets = sorted(
        {
            b
            for b in (
                gcs_bucket_from_path(train_path),
                gcs_bucket_from_path(val_path),
                gcs_bucket_from_path(test_path) if test_path else None,
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
    # Scenario id can be used later to guarantee prediction-map alignment.
    state_features["scenario/id"] = tf.io.FixedLenFeature([], tf.string, default_value=b"")
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
        "future_valid": ex["state/future/valid"],
        "scenario_id": ex["scenario/id"],
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
            future_valid_all = sc.get("future_valid")
            scenario_id = ""
            if "scenario_id" in sc:
                scenario_id_raw = sc["scenario_id"][0].numpy()
                if isinstance(scenario_id_raw, np.ndarray):
                    if scenario_id_raw.shape == ():
                        scenario_id_raw = scenario_id_raw.item()
                    elif scenario_id_raw.size > 0:
                        scenario_id_raw = scenario_id_raw.reshape(-1)[0]
                if isinstance(scenario_id_raw, bytes):
                    scenario_id = scenario_id_raw.decode("utf-8", errors="ignore")
                else:
                    scenario_id = str(scenario_id_raw)

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
            future_xy = fut[sdc_idx, :T, :2].astype(np.float32)
            heading_cos = float(hist[sdc_idx, -1, 4])
            heading_sin = float(hist[sdc_idx, -1, 5])
            anchor_xy = hist[sdc_idx, -1, :2].astype(np.float32)
            if future_valid_all is not None:
                future_valid_np = future_valid_all[0].numpy()[sdc_idx, :T].astype(np.int64)
            else:
                future_valid_np = np.ones((T,), dtype=np.int64)

            self.trajectories.append(
                {
                    "states": states,
                    "actions": actions,
                    "rtg": rtg,
                    "timesteps": np.arange(T, dtype=np.int64),
                    "future_xy": future_xy,
                    "future_valid": future_valid_np,
                    "heading_cos": heading_cos,
                    "heading_sin": heading_sin,
                    "anchor_xy": anchor_xy,
                    "scenario_index": i,
                    "scenario_id": scenario_id,
                }
            )

        if not self.trajectories:
            raise RuntimeError("No trajectories extracted. Check shards/auth/config.")

    def _make_synthetic(self, n: int) -> None:
        """Synthetic circular trajectories for smoke-testing without real data."""
        print(f"No tf_dataset provided — generating {n} synthetic trajectories.")
        T = self.pred_horizon
        for traj_idx in range(n):
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
                    "future_xy": states[:, :2] * POS_SCALE,
                    "future_valid": np.ones((T,), dtype=np.int64),
                    "heading_cos": 1.0,
                    "heading_sin": 0.0,
                    "anchor_xy": np.array([states[0, 0], states[0, 1]], dtype=np.float32) * POS_SCALE,
                    "scenario_index": -1,
                    "scenario_id": f"synthetic_{traj_idx:06d}",
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


# ---------------------------------------------------------------------------
# Training tracker
# ---------------------------------------------------------------------------

@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float
    epoch_secs: float
    samples_trained: int
    peak_gpu_mb: float


@dataclass
class TrainingTracker:
    """
    Tracks wall time and GPU memory across training epochs.

    Usage::

        tracker = TrainingTracker(total_epochs=cfg.epochs, train_loader=train_loader)
        for epoch in range(1, cfg.epochs + 1):
            tracker.start_epoch()
            train_loss = train_one_epoch(...)
            val_loss   = evaluate(...)
            tracker.end_epoch(epoch, train_loss, val_loss)
        tracker.summary()
    """

    total_epochs: int
    train_loader: object          # DataLoader — used to count samples per epoch
    _run_start: float = field(default=0.0, init=False, repr=False)
    _epoch_start: float = field(default=0.0, init=False, repr=False)
    history: list[EpochStats] = field(default_factory=list, init=False)

    def start_run(self) -> None:
        """Call once before the epoch loop."""
        self._reset_gpu_stats()
        self._run_start = time.perf_counter()

    def start_epoch(self) -> None:
        self._reset_gpu_stats()
        self._epoch_start = time.perf_counter()

    def end_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        epoch_secs = time.perf_counter() - self._epoch_start
        elapsed    = time.perf_counter() - self._run_start
        peak_mb    = self._peak_gpu_mb()

        try:
            batch_size = self.train_loader.batch_size or 1
            n_batches  = len(self.train_loader)
            samples    = batch_size * n_batches
        except Exception:
            samples = 0

        stat = EpochStats(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            epoch_secs=epoch_secs,
            samples_trained=samples,
            peak_gpu_mb=peak_mb,
        )
        self.history.append(stat)

        throughput = samples / epoch_secs if epoch_secs > 0 else 0.0
        gpu_info   = f"  peak_gpu={peak_mb:.0f}MB" if peak_mb > 0 else ""
        print(
            f"  elapsed={elapsed:.1f}s  epoch={epoch_secs:.1f}s"
            f"  throughput={throughput:.0f} samples/s{gpu_info}"
        )

    def summary(self) -> None:
        if not self.history:
            return
        total_secs  = sum(s.epoch_secs for s in self.history)
        best        = min(self.history, key=lambda s: s.val_loss)
        peak_mb     = max(s.peak_gpu_mb for s in self.history)
        avg_ep      = total_secs / len(self.history)
        total_samps = sum(s.samples_trained for s in self.history)

        print("\n── Training Summary " + "─" * 41)
        print(f"  Total wall time : {total_secs:.1f}s  ({total_secs/60:.1f} min)")
        print(f"  Avg time/epoch  : {avg_ep:.1f}s")
        print(f"  Total samples   : {total_samps:,}")
        if peak_mb > 0:
            print(f"  Peak GPU memory : {peak_mb:.0f} MB")
        print(
            f"  Best val loss   : {best.val_loss:.5f}  (epoch {best.epoch})"
        )
        print("─" * 60)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _reset_gpu_stats() -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _peak_gpu_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
        return 0.0
