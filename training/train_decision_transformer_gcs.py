from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2Model

from waymo_data_utils import (
    DatasetConfig,
    WOMDOfflineRLDataset,
    build_tf_dataset,
    set_seed,
    validate_gcs_access,
)


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

    if not validate_gcs_access(cfg.train_path, cfg.val_path):
        raise RuntimeError(
            "GCS authentication failed for gs:// data paths. Set "
            "GOOGLE_APPLICATION_CREDENTIALS or run "
            "`gcloud auth application-default login`."
        )

    train_tf = build_tf_dataset(cfg.train_path, cfg.train_shards, cfg.n_agents, cfg.n_past, cfg.n_current, cfg.n_future)
    val_tf = build_tf_dataset(cfg.val_path, cfg.val_shards, cfg.n_agents, cfg.n_past, cfg.n_current, cfg.n_future)
    if train_tf is None:
        raise RuntimeError(f"No train shards found at: {cfg.train_path}")
    if val_tf is None:
        raise RuntimeError(f"No val shards found at: {cfg.val_path}")

    ds_cfg = DatasetConfig(
        state_dim=cfg.state_dim,
        act_dim=cfg.act_dim,
        context_len=cfg.context_len,
        pred_horizon=cfg.pred_horizon,
        rtg_scale=cfg.rtg_scale,
    )
    train_ds = WOMDOfflineRLDataset(train_tf, cfg.max_train_scenarios, ds_cfg)
    val_ds = WOMDOfflineRLDataset(val_tf, cfg.max_val_scenarios, ds_cfg)
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
