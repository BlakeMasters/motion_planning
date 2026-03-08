# Waymo Decision Transformer Workspace

Commit Hygiene 

## Layout

- `training/train_decision_transformer_gcs.py`: main training entrypoint.
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

Convert a local TFRecord shard to torch/npz samples:

```bash
python scripts/convert_waymo_to_torch.py --help
```

Visualize one scenario:

```bash
python scripts/visualize_waymo_sample.py --help
```
