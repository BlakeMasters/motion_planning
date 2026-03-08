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

    # Model.
    hidden_size: int = 128
    n_layer: int = 3
    n_head: int = 1
    dropout: float = 0.1

    # Optimization.
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Runtime.
    seed: int = 42
    num_workers: int = 0
    output_ckpt: str = "outputs/dt_gcs_checkpoint.pt"
    output_config: str = "outputs/dt_gcs_config.json"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a Decision Transformer on Waymo motion data streamed from GCS."
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


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        max_length: int,
        max_ep_len: int,
        n_layer: int,
        n_head: int,
        dropout: float,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        config = GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * hidden_size,
            activation_function="relu",
            n_positions=3 * max_length + 2,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2Model(config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_action = nn.Sequential(nn.Linear(hidden_size, act_dim), nn.Tanh())

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq = states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq), dtype=torch.long, device=states.device)

        t_emb = self.embed_timestep(timesteps)
        s_emb = self.embed_state(states) + t_emb
        a_emb = self.embed_action(actions) + t_emb
        r_emb = self.embed_return(returns_to_go) + t_emb

        stacked = torch.stack([r_emb, s_emb, a_emb], dim=2).reshape(bsz, 3 * seq, self.hidden_size)
        stacked = self.embed_ln(stacked)
        attn = torch.stack([attention_mask] * 3, dim=2).reshape(bsz, 3 * seq)

        out = self.transformer(inputs_embeds=stacked, attention_mask=attn).last_hidden_state
        out = out.reshape(bsz, seq, 3, self.hidden_size)
        return self.predict_action(out[:, :, 1, :])


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        s = batch["states"].to(device)
        a = batch["actions"].to(device)
        rtg = batch["returns_to_go"].to(device)
        ts = batch["timesteps"].to(device)
        msk = batch["attention_mask"].to(device)

        pred = model(s, a, rtg, ts, msk)
        valid = msk.unsqueeze(-1).float()
        loss = (F.mse_loss(pred, a, reduction="none") * valid).sum() / (valid.sum() * model.act_dim + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        s = batch["states"].to(device)
        a = batch["actions"].to(device)
        rtg = batch["returns_to_go"].to(device)
        ts = batch["timesteps"].to(device)
        msk = batch["attention_mask"].to(device)
        pred = model(s, a, rtg, ts, msk)
        valid = msk.unsqueeze(-1).float()
        loss = (F.mse_loss(pred, a, reduction="none") * valid).sum() / (valid.sum() * model.act_dim + 1e-8)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


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

    train_ds = WOMDOfflineRLDataset(train_tf, cfg.max_train_scenarios, cfg)
    val_ds = WOMDOfflineRLDataset(val_tf, cfg.max_val_scenarios, cfg)
    print(f"Train windows: {len(train_ds):,}")
    print(f"Val windows:   {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = DecisionTransformer(
        state_dim=cfg.state_dim,
        act_dim=cfg.act_dim,
        hidden_size=cfg.hidden_size,
        max_length=cfg.context_len,
        max_ep_len=cfg.pred_horizon + 10,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        best_val = min(best_val, val_loss)
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} "
            f"train_mse={train_loss:.5f} val_mse={val_loss:.5f} best_val={best_val:.5f}"
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
        {"model": model.state_dict(), "config": asdict(cfg), "best_val_mse": best_val},
        str(output_ckpt),
    )
    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Saved checkpoint: {output_ckpt}")
    print(f"Saved config:     {output_config}")


if __name__ == "__main__":
    main()
