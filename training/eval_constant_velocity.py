"""
Constant-velocity baseline evaluation.

For each test scenario, extrapolates the SDC's last observed velocity
forward for pred_horizon steps and computes ADE / FDE against the
ground-truth future trajectory.

Usage:
    python eval_constant_velocity.py [--test-shards N] [--max-scenarios N]
                                     [--output outputs/cv_test_predictions.npz]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from waymo_data_utils import (
    DatasetConfig,
    build_tf_dataset,
    validate_gcs_access,
)

DT_S = 0.1          # WOMD timestep (seconds)
GCS_BUCKET = "waymo_open_dataset_motion_v_1_2_0"
TEST_PATH   = f"gs://{GCS_BUCKET}/uncompressed/tf_example/validation"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-shards",    type=int, default=8)
    p.add_argument("--max-scenarios",  type=int, default=100)
    p.add_argument("--output",         type=str,
                   default="outputs/cv_test_predictions.npz")
    p.add_argument("--pred-horizon",   type=int, default=16)
    p.add_argument("--n-agents",       type=int, default=128)
    p.add_argument("--n-past",         type=int, default=10)
    p.add_argument("--n-current",      type=int, default=1)
    p.add_argument("--n-future",       type=int, default=80)
    return p.parse_args()


def main():
    args = parse_args()
    validate_gcs_access(TEST_PATH, TEST_PATH, TEST_PATH)

    ds_cfg = DatasetConfig(
        act_dim=2,
        context_len=20,
        pred_horizon=args.pred_horizon,
        rtg_scale=10.0,
        use_map_features=False,
    )

    test_tf = build_tf_dataset(
        TEST_PATH, args.test_shards,
        args.n_agents, args.n_past, args.n_current, args.n_future,
        use_map_features=False,
    )
    if test_tf is None:
        print("No test shards found."); return

    import tensorflow as tf  # noqa: F401 — needed to iterate tf_dataset

    T = args.pred_horizon
    records = []

    for i, sc in enumerate(test_tf):
        if i >= args.max_scenarios:
            break

        hist = sc["history"][0].numpy()         # (n_agents, n_past+1, 6)
        fut  = sc["future"][0].numpy()           # (n_agents, n_future, 6+)
        sdc  = sc["is_sdc"][0].numpy()

        sdc_idxs = np.where(sdc > 0)[0]
        if len(sdc_idxs) == 0:
            continue
        s = int(sdc_idxs[0])

        # Last history frame: (x, y, vx, vy, cos_h, sin_h)
        last = hist[s, -1]
        anchor_x, anchor_y = last[0], last[1]
        vx, vy = last[2], last[3]

        # CV prediction: linear extrapolation from anchor
        ts = np.arange(1, T + 1, dtype=np.float32) * DT_S   # (T,)
        pred_xy = np.stack([
            anchor_x + vx * ts,
            anchor_y + vy * ts,
        ], axis=-1)                                            # (T, 2)

        # Ground-truth future
        true_xy = fut[s, :T, :2].astype(np.float32)          # (T, 2)

        # Per-step displacement errors
        err = np.linalg.norm(pred_xy - true_xy, axis=-1)     # (T,)
        ade = float(err.mean())
        fde = float(err[-1])

        records.append(dict(
            pred_xy=pred_xy,
            true_xy=true_xy,
            ade_m=ade,
            fde_m=fde,
        ))

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.max_scenarios}  "
                  f"running ADE={np.mean([r['ade_m'] for r in records]):.3f}m")

    pred_xy_all  = np.stack([r["pred_xy"]  for r in records])
    true_xy_all  = np.stack([r["true_xy"]  for r in records])
    ade_all      = np.array([r["ade_m"]    for r in records])
    fde_all      = np.array([r["fde_m"]    for r in records])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), pred_xy=pred_xy_all, true_xy=true_xy_all,
             ade_m=ade_all, fde_m=fde_all)

    print(f"\nConstant-velocity baseline ({len(records)} scenarios)")
    print(f"  ADE = {ade_all.mean():.3f} m")
    print(f"  FDE = {fde_all.mean():.3f} m")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
