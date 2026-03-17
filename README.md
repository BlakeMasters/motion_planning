# Waymo Offline RL with Transformers

Comparative study of Decision Transformer (DT) and Trajectory Transformer (TT) for ego-vehicle motion prediction on the Waymo Open Motion Dataset.

## Layout

- `training/train_decision_transformer_gcs.py` — DT training entrypoint
- `training/train_trajectory_transformer_gcs.py` — TT training entrypoint
- `training/dt_model.py`, `dt_metrics.py`, `dt_trainer.py` — modular DT model/training/metrics
- `training/tt_metrics.py`, `tt_trainer.py` — modular TT training/metrics
- `training/dt_prediction_export.py`, `tt_prediction_export.py` — export test-set predictions (ADE/FDE)
- `training/eval_constant_velocity.py` — constant-velocity baseline evaluation
- `training/waymo_data_utils.py` — shared data pipeline (TFRecord decoding, MDP construction, dataset classes)
- `training/outputs/` — local training artifacts (checkpoints, configs)

## Results summary

Evaluated on 100 held-out WOMD scenarios at a 5s horizon:

| Model | Map | ADE (m) | FDE (m) | Wall time |
|-------|-----|---------|---------|-----------|
| Constant Velocity | — | 2.455 | 6.973 | — |
| DT 500sc | ✗ | 2.240 | 4.235 | 3.06 min |
| TT 500sc | ✗ | 0.781 | 1.685 | 66.3 min |
| DT 500sc | ✓ | 1.596 | 3.305 | 3.19 min |
| TT 500sc | ✓ | 0.748 | 1.812 | 236.5 min |

All transformer models trained on T4 GPU via Google Colab with batch size 64, 50 epochs.

## Common commands

Train Decision Transformer (500 scenarios, with map features):

```bash
python training/train_decision_transformer_gcs.py --max-train-scenarios 500 --use-map-features
```

Train Trajectory Transformer (500 scenarios, no map):

```bash
python training/train_trajectory_transformer_gcs.py --max-train-scenarios 500
```

Convert a local TFRecord shard to torch/npz samples:

```bash
python training/eval_constant_velocity.py
```
