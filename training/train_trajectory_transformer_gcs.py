import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from google.cloud import storage
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2Model


@dataclass
class TrainConfig:
    # Data access.
    gcs_bucket: str = "waymo_open_dataset_motion_v_1_2_0"
    train_path: str = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/training"
    val_path: str = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/validation"
    train_shards: int = 8
    val_shards: int = 2
    max_train_scenarios: int = 500
    max_val_scenarios: int = 100

    # WOMD v1 schema dimensions.
    n_agents: int = 128
    n_past: int = 10
    n_current: int = 1
    n_future: int = 80

    # RL sequence setup.
    state_dim: int = 16
    act_dim: int = 2
    context_len: int = 20
    pred_horizon: int = 16
    rtg_scale: float = 10.0

    # TT-specific.
    n_bins: int = 100         # discretisation bins per dimension
    beam_size: int = 4        # beam width for action inference (not used in training)

    # Model.
    hidden_size: int = 128
    n_layer: int = 3
    n_head: int = 1
    dropout: float = 0.1

    # Optimisation.
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Runtime.
    seed: int = 42
    num_workers: int = 0
    output_ckpt: str = "outputs/tt_gcs_checkpoint.pt"
    output_config: str = "outputs/tt_gcs_config.json"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a Trajectory Transformer on Waymo motion data streamed from GCS."
    )
    parser.add_argument("--gcs-bucket", type=str, default=TrainConfig.gcs_bucket)
    parser.add_argument("--train-path", type=str, default=TrainConfig.train_path)
    parser.add_argument("--val-path", type=str, default=TrainConfig.val_path)
    parser.add_argument("--train-shards", type=int, default=TrainConfig.train_shards)
    parser.add_argument("--val-shards", type=int, default=TrainConfig.val_shards)
    parser.add_argument("--max-train-scenarios", type=int, default=TrainConfig.max_train_scenarios)
    parser.add_argument("--max-val-scenarios", type=int, default=TrainConfig.max_val_scenarios)
    parser.add_argument("--context-len", type=int, default=TrainConfig.context_len)
    parser.add_argument("--pred-horizon", type=int, default=TrainConfig.pred_horizon)
    parser.add_argument("--n-bins", type=int, default=TrainConfig.n_bins)
    parser.add_argument("--beam-size", type=int, default=TrainConfig.beam_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--n-layer", type=int, default=TrainConfig.n_layer)
    parser.add_argument("--n-head", type=int, default=TrainConfig.n_head)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--output-ckpt", type=str, default=TrainConfig.output_ckpt)
    parser.add_argument("--output-config", type=str, default=TrainConfig.output_config)
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.gcs_bucket = args.gcs_bucket
    cfg.train_path = args.train_path
    cfg.val_path = args.val_path
    cfg.train_shards = args.train_shards
    cfg.val_shards = args.val_shards
    cfg.max_train_scenarios = args.max_train_scenarios
    cfg.max_val_scenarios = args.max_val_scenarios
    cfg.context_len = args.context_len
    cfg.pred_horizon = args.pred_horizon
    cfg.n_bins = args.n_bins
    cfg.beam_size = args.beam_size
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.hidden_size = args.hidden_size
    cfg.n_layer = args.n_layer
    cfg.n_head = args.n_head
    cfg.dropout = args.dropout
    cfg.seed = args.seed
    cfg.output_ckpt = args.output_ckpt
    cfg.output_config = args.output_config
    return cfg


# ---------------------------------------------------------------------------
# Shared data utilities (identical to DT script)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def gcs_bucket_from_path(path: str) -> str | None:
    if not path.startswith("gs://"):
        return None
    return path[5:].split("/", 1)[0]


def check_gcs_access(bucket: str) -> bool:
    try:
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


def validate_gcs_access(cfg: TrainConfig) -> bool:
    buckets = sorted(
        {
            b
            for b in (
                gcs_bucket_from_path(cfg.train_path),
                gcs_bucket_from_path(cfg.val_path),
            )
            if b is not None
        }
    )
    if not buckets:
        return True
    return all(check_gcs_access(bucket) for bucket in buckets)


def build_features(cfg: TrainConfig) -> dict[str, tf.io.FixedLenFeature]:
    state_features = {}
    for split, steps in [("past", cfg.n_past), ("current", cfg.n_current), ("future", cfg.n_future)]:
        for feat in ["x", "y", "bbox_yaw", "velocity_x", "velocity_y", "valid"]:
            dtype = tf.int64 if feat == "valid" else tf.float32
            state_features[f"state/{split}/{feat}"] = tf.io.FixedLenFeature(
                [cfg.n_agents, steps], dtype
            )
    for key in ("is_sdc", "type"):
        dtype = tf.int64 if key == "is_sdc" else tf.float32
        state_features[f"state/{key}"] = tf.io.FixedLenFeature([cfg.n_agents], dtype)
    return state_features


def list_shards(path: str, n: int) -> list[str]:
    if path.startswith("gs://"):
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


def parse_scenario(raw: bytes, features: dict[str, tf.io.FixedLenFeature]) -> dict[str, tf.Tensor]:
    ex = tf.io.parse_single_example(raw, features)
    past = tf.stack(
        [
            ex["state/past/x"],
            ex["state/past/y"],
            ex["state/past/velocity_x"],
            ex["state/past/velocity_y"],
            tf.math.cos(ex["state/past/bbox_yaw"]),
            tf.math.sin(ex["state/past/bbox_yaw"]),
        ],
        axis=-1,
    )
    cur = tf.stack(
        [
            ex["state/current/x"],
            ex["state/current/y"],
            ex["state/current/velocity_x"],
            ex["state/current/velocity_y"],
            tf.math.cos(ex["state/current/bbox_yaw"]),
            tf.math.sin(ex["state/current/bbox_yaw"]),
        ],
        axis=-1,
    )
    future = tf.stack(
        [
            ex["state/future/x"],
            ex["state/future/y"],
            ex["state/future/velocity_x"],
            ex["state/future/velocity_y"],
            tf.math.cos(ex["state/future/bbox_yaw"]),
            tf.math.sin(ex["state/future/bbox_yaw"]),
        ],
        axis=-1,
    )
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


def build_tf_dataset(path: str, n_shards: int, cfg: TrainConfig) -> tf.data.Dataset | None:
    shards = list_shards(path, n_shards)
    if not shards:
        return None
    print(f"Using {len(shards)} shards from {path}")
    print(f"First shard: {shards[0]}")
    features = build_features(cfg)
    ds = (
        tf.data.TFRecordDataset(shards, num_parallel_reads=4)
        .map(lambda raw: parse_scenario(raw, features), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(1)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds


POS_SCALE = 50.0
VEL_SCALE = 15.0
ACT_SCALE = 1.5
K_NEAREST = 5


def build_state(hist: np.ndarray, sdc_idx: int) -> np.ndarray:
    t = hist.shape[1] - 1
    ego = hist[sdc_idx, t]
    ex, ey, evx, evy, cosh, sinh = ego

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
    if t == 0:
        return np.zeros((2,), dtype=np.float32)
    dx = (fut[sdc_idx, t, 0] - fut[sdc_idx, t - 1, 0]) / ACT_SCALE
    dy = (fut[sdc_idx, t, 1] - fut[sdc_idx, t - 1, 1]) / ACT_SCALE
    cos_h = hist[sdc_idx, -1, 4]
    sin_h = hist[sdc_idx, -1, 5]
    xl = dx * cos_h + dy * sin_h
    yl = -dx * sin_h + dy * cos_h
    return np.array([xl, yl], dtype=np.float32)


def compute_rtg(fut: np.ndarray, sdc_idx: int, horizon: int, rtg_scale: float) -> np.ndarray:
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


class WOMDOfflineRLDataset(Dataset):
    """Builds continuous (s, a, rtg) trajectories from WOMD TFRecord data."""

    def __init__(self, tf_dataset: tf.data.Dataset, max_scenarios: int, cfg: TrainConfig):
        self.context_len = cfg.context_len
        self.pred_horizon = cfg.pred_horizon
        self.state_dim = cfg.state_dim
        self.act_dim = cfg.act_dim
        self.windows_per_traj = max(1, self.pred_horizon - self.context_len + 1)
        self.trajectories: list[dict[str, np.ndarray]] = []

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
            states = np.array([build_state(hist, sdc_idx) for _ in range(T)], dtype=np.float32)
            actions = np.array([build_action(fut, hist, sdc_idx, t) for t in range(T)], dtype=np.float32)
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
            return np.concatenate([arr[start:T], np.zeros((pad,) + arr.shape[1:], dtype=arr.dtype)], axis=0)

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
# TT-specific dataset: tokenise continuous (s, a) into flat discrete tokens
# ---------------------------------------------------------------------------

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

    def __init__(self, rl_dataset: WOMDOfflineRLDataset, model: "TrajectoryTransformer"):
        self.rl_dataset = rl_dataset
        self.model = model

    def __len__(self) -> int:
        return len(self.rl_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.rl_dataset[idx]
        states = sample["states"].numpy()    # (CL, state_dim)
        actions = sample["actions"].numpy()  # (CL, act_dim)
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
        mask_list.append(torch.cat([torch.ones(L, dtype=torch.long),
                                    torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


# ---------------------------------------------------------------------------
# Trajectory Transformer model
# ---------------------------------------------------------------------------

class TrajectoryTransformer(nn.Module):
    """
    GPT-2 based next-token predictor over discretised state-action-reward
    sequences (Janner et al., 2021).

    Flat vocabulary layout
    ----------------------
    state  dim d  →  tokens  [d*B,          (d+1)*B)
    action dim d  →  tokens  [(state_dim+d)*B, (state_dim+d+1)*B)
    reward        →  one placeholder token at (state_dim + act_dim) * B

    vocab_size = (state_dim + act_dim + 1) * n_bins  (reward gets 1 token)
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        n_bins: int,
        hidden_size: int,
        n_layer: int,
        n_head: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.n_bins = n_bins
        self.tokens_per_step = state_dim + act_dim + 1

        # Token ID ranges for each segment type.
        self.state_offset = 0
        self.action_offset = state_dim * n_bins
        self.reward_token = (state_dim + act_dim) * n_bins   # single placeholder
        self.vocab_size = self.reward_token + 1

        config = GPT2Config(
            vocab_size=self.vocab_size,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * hidden_size,
            n_positions=max_seq_len,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            activation_function="gelu",
        )
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(hidden_size, self.vocab_size, bias=False)
        # Weight tying: share embedding and output projection (standard LM practice).
        self.transformer.wte.weight = self.lm_head.weight

        # Per-dimension percentile bin edges (fitted from data, not learned).
        self._state_bins: list[np.ndarray] | None = None
        self._action_bins: list[np.ndarray] | None = None

    # -- Discretisation -------------------------------------------------------

    def fit_discretisation(self, states_all: np.ndarray, actions_all: np.ndarray) -> None:
        """
        Fit per-dimension percentile bin edges on collected training data.

        states_all  : (N, state_dim)
        actions_all : (N, act_dim)
        """
        def _edges(data: np.ndarray, n_dims: int) -> list[np.ndarray]:
            result = []
            for d in range(n_dims):
                col = data[:, d]
                q = np.quantile(col, np.linspace(0, 1, self.n_bins + 1))
                q = np.unique(q)  # deduplicate (can happen with constant dims)
                result.append(q)
            return result

        self._state_bins = _edges(states_all, self.state_dim)
        self._action_bins = _edges(actions_all, self.act_dim)
        print(f"  Discretisation fitted on {len(states_all):,} state samples")

    def _discretise(self, val: float, edges: np.ndarray) -> int:
        """Map a scalar to a bin index in [0, n_bins-1]."""
        idx = int(np.searchsorted(edges, val, side="right")) - 1
        return int(np.clip(idx, 0, len(edges) - 2))

    def tokenise(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Convert one context window to a flat integer token sequence.

        states  : (T, state_dim)
        actions : (T, act_dim)
        Returns : (T * tokens_per_step,)  int64
        """
        assert self._state_bins is not None, "Call fit_discretisation first."
        T = states.shape[0]
        tokens: list[int] = []
        for t in range(T):
            for d in range(self.state_dim):
                idx = self._discretise(states[t, d], self._state_bins[d])
                tokens.append(self.state_offset + d * self.n_bins + idx)
            for d in range(self.act_dim):
                idx = self._discretise(actions[t, d], self._action_bins[d])
                tokens.append(self.action_offset + d * self.n_bins + idx)
            tokens.append(self.reward_token)
        return np.array(tokens, dtype=np.int64)

    def decode_action_token(self, token_id: int, dim: int) -> float:
        """Decode a single action token back to a continuous value (bin centre)."""
        assert self._action_bins is not None
        local_idx = token_id - (self.action_offset + dim * self.n_bins)
        local_idx = int(np.clip(local_idx, 0, self.n_bins - 1))
        edges = self._action_bins[dim]
        if local_idx + 1 < len(edges):
            return float(0.5 * (edges[local_idx] + edges[local_idx + 1]))
        return float(edges[-1])

    # -- Forward --------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        token_ids      : (B, L)  long
        attention_mask : (B, L)  long  (1 = real, 0 = pad)
        Returns logits : (B, L, vocab_size)
        """
        hidden = self.transformer(
            input_ids=token_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.lm_head(hidden)

    # -- Beam-search action inference -----------------------------------------

    @torch.no_grad()
    def beam_search_actions(
        self,
        context_tokens: torch.Tensor,
        beam_size: int = 4,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Given a context token sequence ending just before action tokens,
        beam-search the most likely action tokens for the next timestep.

        context_tokens : (L,) long  — full context including current state tokens
        Returns        : (act_dim,) long  — best-beam action token per dimension
        """
        self.eval()
        max_ctx = self.transformer.config.n_positions - self.act_dim - 2
        if context_tokens.shape[0] > max_ctx:
            context_tokens = context_tokens[-max_ctx:]

        if device is not None:
            context_tokens = context_tokens.to(device)
        context = context_tokens.unsqueeze(0)   # (1, L)

        beams: list[tuple[float, torch.Tensor]] = [(0.0, context)]

        for step in range(self.act_dim):
            new_beams = []
            for score, seq in beams:
                logits = self.forward(seq)[:, -1, :]   # (1, vocab_size)

                # Restrict to the n_bins tokens for this action dimension.
                act_start = self.action_offset + step * self.n_bins
                act_logits = logits[0, act_start: act_start + self.n_bins]
                log_probs = F.log_softmax(act_logits, dim=-1)

                topk_vals, topk_idx = log_probs.topk(min(beam_size, self.n_bins))
                for log_p, local_idx in zip(topk_vals, topk_idx):
                    tok = torch.tensor(
                        [[act_start + local_idx.item()]], dtype=torch.long,
                        device=seq.device,
                    )
                    new_seq = torch.cat([seq, tok], dim=1)
                    new_beams.append((score + log_p.item(), new_seq))

            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]

        best_seq = beams[0][1][0]              # (L + act_dim,)
        return best_seq[-self.act_dim:]        # (act_dim,)


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: TrajectoryTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        lbls = batch["labels"].to(device)
        msk = batch["attention_mask"].to(device)

        logits = model(ids, attention_mask=msk)          # (B, L, V)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            lbls.reshape(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: TrajectoryTransformer,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        lbls = batch["labels"].to(device)
        msk = batch["attention_mask"].to(device)

        logits = model(ids, attention_mask=msk)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            lbls.reshape(-1),
            ignore_index=-100,
        )
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not validate_gcs_access(cfg):
        raise RuntimeError(
            "GCS authentication failed for gs:// data paths. Set "
            "GOOGLE_APPLICATION_CREDENTIALS or run "
            "`gcloud auth application-default login`."
        )

    train_tf = build_tf_dataset(cfg.train_path, cfg.train_shards, cfg)
    val_tf = build_tf_dataset(cfg.val_path, cfg.val_shards, cfg)
    if train_tf is None:
        raise RuntimeError(f"No train shards found at: {cfg.train_path}")
    if val_tf is None:
        raise RuntimeError(f"No val shards found at: {cfg.val_path}")

    # Build continuous RL datasets (same pipeline as DT).
    train_rl = WOMDOfflineRLDataset(train_tf, cfg.max_train_scenarios, cfg)
    val_rl = WOMDOfflineRLDataset(val_tf, cfg.max_val_scenarios, cfg)
    print(f"Train windows: {len(train_rl):,}")
    print(f"Val windows:   {len(val_rl):,}")

    # Tokens per step: state_dim + act_dim + 1 (reward placeholder).
    tokens_per_step = cfg.state_dim + cfg.act_dim + 1
    # Max sequence length: context_len steps, each tokenised, minus 1 for LM shift.
    max_seq_len = cfg.context_len * tokens_per_step + 16   # +16 headroom

    model = TrajectoryTransformer(
        state_dim=cfg.state_dim,
        act_dim=cfg.act_dim,
        n_bins=cfg.n_bins,
        hidden_size=cfg.hidden_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
        max_seq_len=max_seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trajectory Transformer | params: {n_params:,}")
    print(f"Vocabulary size: {model.vocab_size}  "
          f"(state_dim={cfg.state_dim}, act_dim={cfg.act_dim}, n_bins={cfg.n_bins})")
    print(f"Tokens per step: {tokens_per_step}  |  max_seq_len: {max_seq_len}")

    # Fit discretisation bin edges on training data.
    print("Fitting per-dimension discretisation bin edges on training data...")
    all_states = np.concatenate(
        [train_rl[i]["states"].numpy() for i in range(len(train_rl))], axis=0
    )
    all_actions = np.concatenate(
        [train_rl[i]["actions"].numpy() for i in range(len(train_rl))], axis=0
    )
    model.fit_discretisation(all_states, all_actions)

    # Wrap in TT token datasets.
    train_tt = TTTokenDataset(train_rl, model)
    val_tt = TTTokenDataset(val_rl, model)

    train_loader = DataLoader(
        train_tt,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=tt_collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_tt,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=tt_collate,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        best_val = min(best_val, val_loss)
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} "
            f"train_ce={train_loss:.5f} val_ce={val_loss:.5f} best_val={best_val:.5f}"
        )

    script_dir = Path(__file__).resolve().parent
    output_ckpt = Path(cfg.output_ckpt)
    output_config = Path(cfg.output_config)
    if not output_ckpt.is_absolute():
        output_ckpt = script_dir / output_ckpt
    if not output_config.is_absolute():
        output_config = script_dir / output_config
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    output_config.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "best_val_ce": best_val,
            # Save bin edges so the checkpoint is self-contained for inference.
            "state_bins": model._state_bins,
            "action_bins": model._action_bins,
        },
        str(output_ckpt),
    )
    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Saved checkpoint: {output_ckpt}")
    print(f"Saved config:     {output_config}")


if __name__ == "__main__":
    main()
