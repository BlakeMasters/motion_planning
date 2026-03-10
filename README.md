# Waymo Decision Transformer Workspace

Commit Hygiene 

## Layout

- `training/train_decision_transformer_gcs.py`: main training entrypoint.
- `training/dt_model.py`, `training/dt_metrics.py`, `training/dt_trainer.py`: modular DT model/training/metrics code.
- `training/dt_prediction_export.py`: exports test-set DT predictions + ADE/FDE for visualization.
- `training/outputs/`: default training artifacts (checkpoint/config).
- `scripts/`: utility scripts for conversion, inspection, visualization, and notebook-exported code.
- `notebooks/`: top-level exploratory notebooks.
- `docs/`: notes/reference text files.
- `tutorial/`: Waymo tutorial assets and notebooks.

## Common commands

Train Decision Transformer from GCS:

```bash
python training/train_decision_transformer_gcs.py
```

Small run with explicit sample controls:

```bash
python training/train_decision_transformer_gcs.py --train-shards 1 --val-shards 1 --test-shards 1 --max-train-scenarios 20 --max-val-scenarios 10 --max-test-scenarios 10 --max-train-samples 200 --max-val-samples 100 --max-test-samples 100 --epochs 1
```

Convert a local TFRecord shard to torch/npz samples:

```bash
python scripts/convert_waymo_to_torch.py --help
```

Visualize one scenario:

```bash
python scripts/visualize_waymo_sample.py --help
```

Overlay Decision Transformer predictions vs true trajectory on map:

```bash
python scripts/visualize_waymo_sample.py --tfrecord uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000 --predictions training/outputs/dt_test_predictions.npz --prediction-index 0 --output dt_overlay.png
```
