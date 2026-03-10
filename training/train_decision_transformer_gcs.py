from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dt_model import DecisionTransformer
from dt_prediction_export import (
    generate_sample_predictions,
    save_sample_predictions,
    summarise_sample_predictions,
)
from dt_trainer import evaluate, train_one_epoch
from waymo_data_utils import (
    DatasetConfig,
    TrainingTracker,
    WOMDOfflineRLDataset,
    build_tf_dataset,
    save_training_plot,
    set_seed,
    upload_checkpoint,
    validate_gcs_access,
)


@dataclass
class TrainConfig:
    # Data access.
    gcs_bucket: str = "waymo_open_dataset_motion_v_1_2_0"
    train_path: str = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/training"
    val_path: str = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/validation"
    test_path: str = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/validation"
    train_shards: int = 8
    val_shards: int = 2
    test_shards: int = 2
    max_train_scenarios: int = 500
    max_val_scenarios: int = 100
    max_test_scenarios: int = 100
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    max_test_predictions: int = 20

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

    # Runtime/artifacts.
    seed: int = 42
    num_workers: int = 0
    output_ckpt: str = "outputs/dt_gcs_checkpoint.pt"
    output_config: str = "outputs/dt_gcs_config.json"
    output_metrics: str = "outputs/dt_gcs_metrics.json"
    output_test_predictions: str = "outputs/dt_test_predictions.npz"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a Decision Transformer on Waymo motion data streamed from GCS."
    )
    parser.add_argument("--gcs-bucket", type=str, default=TrainConfig.gcs_bucket)
    parser.add_argument("--train-path", type=str, default=TrainConfig.train_path)
    parser.add_argument("--val-path", type=str, default=TrainConfig.val_path)
    parser.add_argument("--test-path", type=str, default=TrainConfig.test_path)
    parser.add_argument("--train-shards", type=int, default=TrainConfig.train_shards)
    parser.add_argument("--val-shards", type=int, default=TrainConfig.val_shards)
    parser.add_argument("--test-shards", type=int, default=TrainConfig.test_shards)
    parser.add_argument("--max-train-scenarios", type=int, default=TrainConfig.max_train_scenarios)
    parser.add_argument("--max-val-scenarios", type=int, default=TrainConfig.max_val_scenarios)
    parser.add_argument("--max-test-scenarios", type=int, default=TrainConfig.max_test_scenarios)
    parser.add_argument("--max-train-samples", type=int, default=TrainConfig.max_train_samples)
    parser.add_argument("--max-val-samples", type=int, default=TrainConfig.max_val_samples)
    parser.add_argument("--max-test-samples", type=int, default=TrainConfig.max_test_samples)
    parser.add_argument("--max-test-predictions", type=int, default=TrainConfig.max_test_predictions)
    parser.add_argument("--context-len", type=int, default=TrainConfig.context_len)
    parser.add_argument("--pred-horizon", type=int, default=TrainConfig.pred_horizon)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--n-layer", type=int, default=TrainConfig.n_layer)
    parser.add_argument("--n-head", type=int, default=TrainConfig.n_head)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--output-ckpt", type=str, default=TrainConfig.output_ckpt)
    parser.add_argument("--output-config", type=str, default=TrainConfig.output_config)
    parser.add_argument("--output-metrics", type=str, default=TrainConfig.output_metrics)
    parser.add_argument("--output-test-predictions", type=str, default=TrainConfig.output_test_predictions)
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.gcs_bucket = args.gcs_bucket
    cfg.train_path = args.train_path
    cfg.val_path = args.val_path
    cfg.test_path = args.test_path
    cfg.train_shards = args.train_shards
    cfg.val_shards = args.val_shards
    cfg.test_shards = args.test_shards
    cfg.max_train_scenarios = args.max_train_scenarios
    cfg.max_val_scenarios = args.max_val_scenarios
    cfg.max_test_scenarios = args.max_test_scenarios
    cfg.max_train_samples = args.max_train_samples
    cfg.max_val_samples = args.max_val_samples
    cfg.max_test_samples = args.max_test_samples
    cfg.max_test_predictions = args.max_test_predictions
    cfg.context_len = args.context_len
    cfg.pred_horizon = args.pred_horizon
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.hidden_size = args.hidden_size
    cfg.n_layer = args.n_layer
    cfg.n_head = args.n_head
    cfg.dropout = args.dropout
    cfg.seed = args.seed
    cfg.output_ckpt = args.output_ckpt
    cfg.output_config = args.output_config
    cfg.output_metrics = args.output_metrics
    cfg.output_test_predictions = args.output_test_predictions
    return cfg


def _resolve_output_path(script_dir: Path, output_path: str) -> Path:
    path = Path(output_path)
    return path if path.is_absolute() else script_dir / path


def _subset_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset), dtype=np.int64)
    rng.shuffle(indices)
    return Subset(dataset, indices[:max_samples].tolist())


def _build_rl_dataset(
    path: str,
    shards: int,
    max_scenarios: int,
    cfg: TrainConfig,
) -> WOMDOfflineRLDataset:
    tf_ds = build_tf_dataset(path, shards, cfg.n_agents, cfg.n_past, cfg.n_current, cfg.n_future)
    if tf_ds is None:
        raise RuntimeError(f"No shards found at: {path}")
    ds_cfg = DatasetConfig(
        state_dim=cfg.state_dim,
        act_dim=cfg.act_dim,
        context_len=cfg.context_len,
        pred_horizon=cfg.pred_horizon,
        rtg_scale=cfg.rtg_scale,
    )
    return WOMDOfflineRLDataset(tf_ds, max_scenarios, ds_cfg)


def _make_loader(dataset: Dataset, cfg: TrainConfig, shuffle: bool, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not validate_gcs_access(cfg.train_path, cfg.val_path, cfg.test_path):
        raise RuntimeError(
            "GCS authentication failed for gs:// data paths. Set "
            "GOOGLE_APPLICATION_CREDENTIALS or run "
            "`gcloud auth application-default login`."
        )

    train_ds = _build_rl_dataset(cfg.train_path, cfg.train_shards, cfg.max_train_scenarios, cfg)
    val_ds = _build_rl_dataset(cfg.val_path, cfg.val_shards, cfg.max_val_scenarios, cfg)
    test_ds = _build_rl_dataset(cfg.test_path, cfg.test_shards, cfg.max_test_scenarios, cfg)

    train_ds_for_loader = _subset_dataset(train_ds, cfg.max_train_samples, cfg.seed + 1)
    val_ds_for_loader = _subset_dataset(val_ds, cfg.max_val_samples, cfg.seed + 2)
    test_ds_for_loader = _subset_dataset(test_ds, cfg.max_test_samples, cfg.seed + 3)

    print(
        "Train windows: "
        f"{len(train_ds_for_loader):,}/{len(train_ds):,} "
        f"(sample_limit={cfg.max_train_samples})"
    )
    print(
        "Val windows:   "
        f"{len(val_ds_for_loader):,}/{len(val_ds):,} "
        f"(sample_limit={cfg.max_val_samples})"
    )
    print(
        "Test windows:  "
        f"{len(test_ds_for_loader):,}/{len(test_ds):,} "
        f"(sample_limit={cfg.max_test_samples})"
    )

    train_loader = _make_loader(train_ds_for_loader, cfg, shuffle=True, pin_memory=(device.type == "cuda"))
    val_loader = _make_loader(val_ds_for_loader, cfg, shuffle=False, pin_memory=False)
    test_loader = _make_loader(test_ds_for_loader, cfg, shuffle=False, pin_memory=False)

    model = DecisionTransformer(
        state_dim=cfg.state_dim,
        act_dim=cfg.act_dim,
        hidden_size=cfg.hidden_size,
        max_length=cfg.context_len,
        max_ep_len=max(cfg.pred_horizon + cfg.context_len + 2, cfg.pred_horizon + 10),
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    tracker = TrainingTracker(total_epochs=cfg.epochs, train_loader=train_loader)
    tracker.start_run()
    best_val_mse = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    epoch_history: list[dict[str, dict[str, float] | int]] = []

    for epoch in range(1, cfg.epochs + 1):
        tracker.start_epoch()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, grad_clip=cfg.grad_clip)
        val_metrics = evaluate(model, val_loader, device)
        improved = val_metrics.action_mse < best_val_mse
        if improved:
            best_val_mse = val_metrics.action_mse
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        epoch_history.append(
            {
                "epoch": epoch,
                "train": train_metrics.to_dict(),
                "val": val_metrics.to_dict(),
            }
        )
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} "
            f"train_mse={train_metrics.action_mse:.5f} train_rmse={train_metrics.action_rmse:.5f} "
            f"train_ade={train_metrics.ade_m:.3f} train_fde={train_metrics.fde_m:.3f} | "
            f"val_mse={val_metrics.action_mse:.5f} val_rmse={val_metrics.action_rmse:.5f} "
            f"val_ade={val_metrics.ade_m:.3f} val_fde={val_metrics.fde_m:.3f} "
            f"{'(best)' if improved else ''}"
        )
        tracker.end_epoch(epoch, train_metrics.action_mse, val_metrics.action_mse)
    tracker.summary()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_metrics = evaluate(model, test_loader, device)
    print(
        "Test metrics | "
        f"mse={test_metrics.action_mse:.5f} "
        f"mae={test_metrics.action_mae:.5f} "
        f"rmse={test_metrics.action_rmse:.5f} "
        f"ade={test_metrics.ade_m:.3f}m "
        f"fde={test_metrics.fde_m:.3f}m"
    )

    test_prediction_summary: dict[str, float] = {"num_samples": 0}
    test_prediction_samples = []
    if cfg.max_test_predictions > 0:
        test_prediction_samples = generate_sample_predictions(
            model=model,
            dataset=test_ds,
            device=device,
            max_samples=cfg.max_test_predictions,
        )
        test_prediction_summary = summarise_sample_predictions(test_prediction_samples)
        print(
            "Saved-sample metrics | "
            f"count={int(test_prediction_summary['num_samples'])} "
            f"ade_mean={test_prediction_summary.get('ade_mean_m', float('nan')):.3f}m "
            f"fde_mean={test_prediction_summary.get('fde_mean_m', float('nan')):.3f}m"
        )

    script_dir = Path(__file__).resolve().parent
    output_ckpt = _resolve_output_path(script_dir, cfg.output_ckpt)
    output_config = _resolve_output_path(script_dir, cfg.output_config)
    output_metrics = _resolve_output_path(script_dir, cfg.output_metrics)
    output_test_predictions = _resolve_output_path(script_dir, cfg.output_test_predictions)

    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_test_predictions.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "best_val_action_mse": best_val_mse,
            "test_metrics": test_metrics.to_dict(),
            "test_sample_metrics": test_prediction_summary,
        },
        str(output_ckpt),
    )

    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    payload = {
        "best_val_action_mse": best_val_mse,
        "test_metrics": test_metrics.to_dict(),
        "test_sample_metrics": test_prediction_summary,
        "epochs": epoch_history,
    }
    with open(output_metrics, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if test_prediction_samples:
        save_sample_predictions(output_test_predictions, test_prediction_samples)
        print(f"Saved test predictions: {output_test_predictions}")

    print(f"Saved checkpoint:       {output_ckpt}")
    print(f"Saved config:           {output_config}")
    print(f"Saved metric report:    {output_metrics}")

    train_losses = [m["train"]["action_mse"] for m in epoch_history]
    val_losses = [m["val"]["action_mse"] for m in epoch_history]
    plot_path = str(output_ckpt.parent / "dt_training_curves.png")
    save_training_plot(train_losses, val_losses, "MSE Loss", plot_path)
    upload_checkpoint(str(output_ckpt))
    upload_checkpoint(str(output_config))


if __name__ == "__main__":
    main()
