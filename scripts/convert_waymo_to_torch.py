import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import scenario_pb2


DEFAULT_TFRECORD = (
    "uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000"
)


@dataclass
class Episode:
    scenario_id: str
    track_id: int
    object_type: int
    states: np.ndarray  # [T, state_dim]
    actions: np.ndarray  # [T, action_dim]
    rewards: np.ndarray  # [T]
    timesteps: np.ndarray  # [T]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Waymo Scenario TFRecords into PyTorch-ready DT/TT datasets."
    )
    parser.add_argument("--tfrecord", type=str, default=DEFAULT_TFRECORD)
    parser.add_argument("--output-dir", type=str, default="torch_data")
    parser.add_argument("--prefix", type=str, default="waymo")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--min-valid-steps", type=int, default=12)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--tt-discretize",
        action="store_true",
        help="If set, output integer TT tokens instead of float features.",
    )
    parser.add_argument("--tt-num-bins", type=int, default=256)
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["pt", "npz"],
        default="pt",
        help="Dataset artifact format. Use 'pt' for PyTorch, 'npz' if torch is unavailable.",
    )
    return parser.parse_args()


def load_scenarios(path: str, max_scenarios: int | None = None):
    dataset = tf.data.TFRecordDataset(path, compression_type="")
    for idx, raw_record in enumerate(dataset):
        if max_scenarios is not None and idx >= max_scenarios:
            break
        yield scenario_pb2.Scenario.FromString(raw_record.numpy())


def state_from_track(track: scenario_pb2.Track, state_idx: int) -> np.ndarray:
    s = track.states[state_idx]
    return np.array(
        [
            s.center_x,
            s.center_y,
            s.velocity_x,
            s.velocity_y,
            s.heading,
            s.length,
            s.width,
            s.height,
        ],
        dtype=np.float32,
    )


def action_from_track(
    track: scenario_pb2.Track, state_idx: int, next_state_idx: int
) -> np.ndarray:
    s = track.states[state_idx]
    n = track.states[next_state_idx]
    return np.array(
        [
            n.center_x - s.center_x,
            n.center_y - s.center_y,
            n.heading - s.heading,
        ],
        dtype=np.float32,
    )


def discounted_return_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        rtg[i] = running
    return rtg


def scenario_to_episodes(
    scenario: scenario_pb2.Scenario, min_valid_steps: int
) -> list[Episode]:
    episodes: list[Episode] = []
    for track in scenario.tracks:
        valid_idx = [i for i, s in enumerate(track.states) if s.valid]
        if len(valid_idx) < min_valid_steps:
            continue

        states = []
        actions = []
        rewards = []
        timesteps = []

        for i in range(len(valid_idx) - 1):
            cur_idx = valid_idx[i]
            nxt_idx = valid_idx[i + 1]
            state = state_from_track(track, cur_idx)
            action = action_from_track(track, cur_idx, nxt_idx)
            reward = float(np.linalg.norm(action[:2]))

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            timesteps.append(cur_idx)

        if not states:
            continue

        episodes.append(
            Episode(
                scenario_id=scenario.scenario_id,
                track_id=track.id,
                object_type=track.object_type,
                states=np.stack(states, axis=0),
                actions=np.stack(actions, axis=0),
                rewards=np.array(rewards, dtype=np.float32),
                timesteps=np.array(timesteps, dtype=np.int64),
            )
        )
    return episodes


def build_dt_dataset(
    episodes: list[Episode], context_len: int, stride: int, gamma: float
) -> dict[str, np.ndarray]:
    state_dim = episodes[0].states.shape[-1]
    action_dim = episodes[0].actions.shape[-1]

    states_out = []
    actions_out = []
    rewards_out = []
    rtg_out = []
    timesteps_out = []
    mask_out = []
    object_type_out = []

    for ep in episodes:
        ep_rtg = discounted_return_to_go(ep.rewards, gamma)
        T = ep.states.shape[0]
        for start in range(0, T, stride):
            end = min(start + context_len, T)
            span = end - start
            if span <= 0:
                continue

            s = np.zeros((context_len, state_dim), dtype=np.float32)
            a = np.zeros((context_len, action_dim), dtype=np.float32)
            r = np.zeros((context_len,), dtype=np.float32)
            g = np.zeros((context_len,), dtype=np.float32)
            t = np.zeros((context_len,), dtype=np.int64)
            m = np.zeros((context_len,), dtype=np.float32)

            s[-span:] = ep.states[start:end]
            a[-span:] = ep.actions[start:end]
            r[-span:] = ep.rewards[start:end]
            g[-span:] = ep_rtg[start:end]
            t[-span:] = ep.timesteps[start:end]
            m[-span:] = 1.0

            states_out.append(s)
            actions_out.append(a)
            rewards_out.append(r)
            rtg_out.append(g)
            timesteps_out.append(t)
            mask_out.append(m)
            object_type_out.append(ep.object_type)

    return {
        "states": np.stack(states_out, axis=0),
        "actions": np.stack(actions_out, axis=0),
        "rewards": np.stack(rewards_out, axis=0),
        "returns_to_go": np.stack(rtg_out, axis=0),
        "timesteps": np.stack(timesteps_out, axis=0),
        "attention_mask": np.stack(mask_out, axis=0),
        "object_type": np.array(object_type_out, dtype=np.int64),
    }


def quantize_features(
    features: np.ndarray, num_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    fmin = flat.min(axis=0)
    fmax = flat.max(axis=0)
    denom = np.maximum(fmax - fmin, 1e-6)
    normalized = (features - fmin) / denom
    quantized = np.clip(np.floor(normalized * (num_bins - 1)), 0, num_bins - 1)
    return quantized.astype(np.int64), fmin.astype(np.float32), fmax.astype(np.float32)


def build_tt_dataset(
    episodes: list[Episode], context_len: int, stride: int, discretize: bool, num_bins: int
) -> dict[str, np.ndarray]:
    # Per-step TT feature vector: [state(8), action(3), reward(1)].
    feature_dim = episodes[0].states.shape[-1] + episodes[0].actions.shape[-1] + 1
    sequences = []
    mask_out = []
    object_type_out = []

    for ep in episodes:
        step_features = np.concatenate(
            [ep.states, ep.actions, ep.rewards[:, None]], axis=-1
        ).astype(np.float32)
        T = step_features.shape[0]
        for start in range(0, T, stride):
            end = min(start + context_len, T)
            span = end - start
            if span <= 0:
                continue

            seq = np.zeros((context_len, feature_dim), dtype=np.float32)
            mask = np.zeros((context_len,), dtype=np.float32)
            seq[-span:] = step_features[start:end]
            mask[-span:] = 1.0

            sequences.append(seq)
            mask_out.append(mask)
            object_type_out.append(ep.object_type)

    seq_np = np.stack(sequences, axis=0)
    out: dict[str, np.ndarray] = {
        "attention_mask": np.stack(mask_out, axis=0),
        "object_type": np.array(object_type_out, dtype=np.int64),
    }

    if discretize:
        tokens, fmin, fmax = quantize_features(seq_np, num_bins)
        out["tokens"] = tokens
        out["feature_min"] = fmin
        out["feature_max"] = fmax
        out["num_bins"] = np.array([num_bins], dtype=np.int64)
    else:
        out["features"] = seq_np

    return out


def convert_numpy_to_torch(obj):
    import torch

    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy_to_torch(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_torch(v) for v in obj]
    return obj


def save_pt(path: Path, payload: dict) -> None:
    import torch

    torch.save(convert_numpy_to_torch(payload), path)


def save_npz(path: Path, payload: dict) -> None:
    np.savez_compressed(path, **payload)


def main() -> None:
    args = parse_args()
    tfrecord_path = Path(args.tfrecord)
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes: list[Episode] = []
    n_scenarios = 0
    for scenario in load_scenarios(str(tfrecord_path), args.max_scenarios):
        n_scenarios += 1
        episodes.extend(scenario_to_episodes(scenario, args.min_valid_steps))

    if not episodes:
        raise RuntimeError(
            "No episodes extracted. Lower --min-valid-steps or increase --max-scenarios."
        )

    dt_data = build_dt_dataset(episodes, args.context_len, args.stride, args.gamma)
    tt_data = build_tt_dataset(
        episodes,
        args.context_len,
        args.stride,
        args.tt_discretize,
        args.tt_num_bins,
    )

    episodes_payload = {
        "scenario_id": np.array([ep.scenario_id for ep in episodes]),
        "track_id": np.array([ep.track_id for ep in episodes], dtype=np.int64),
        "object_type": np.array([ep.object_type for ep in episodes], dtype=np.int64),
        "states": [ep.states for ep in episodes],
        "actions": [ep.actions for ep in episodes],
        "rewards": [ep.rewards for ep in episodes],
        "timesteps": [ep.timesteps for ep in episodes],
    }

    if args.output_format == "pt":
        try:
            save_pt(out_dir / f"{args.prefix}_episodes.pt", episodes_payload)
            save_pt(out_dir / f"{args.prefix}_dt.pt", dt_data)
            save_pt(out_dir / f"{args.prefix}_tt.pt", tt_data)
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise ModuleNotFoundError(
                    "PyTorch is not installed. Install torch or rerun with "
                    "--output-format npz."
                ) from exc
            raise
    else:
        # Ragged episode tensors are stored as object arrays in the npz fallback.
        episodes_npz = {
            "scenario_id": episodes_payload["scenario_id"],
            "track_id": episodes_payload["track_id"],
            "object_type": episodes_payload["object_type"],
            "states": np.array(episodes_payload["states"], dtype=object),
            "actions": np.array(episodes_payload["actions"], dtype=object),
            "rewards": np.array(episodes_payload["rewards"], dtype=object),
            "timesteps": np.array(episodes_payload["timesteps"], dtype=object),
        }
        save_npz(out_dir / f"{args.prefix}_episodes.npz", episodes_npz)
        save_npz(out_dir / f"{args.prefix}_dt.npz", dt_data)
        save_npz(out_dir / f"{args.prefix}_tt.npz", tt_data)

    metadata = {
        "tfrecord": str(tfrecord_path),
        "n_scenarios": n_scenarios,
        "n_episodes": len(episodes),
        "dt_num_samples": int(dt_data["states"].shape[0]),
        "tt_num_samples": int(
            tt_data["tokens"].shape[0]
            if "tokens" in tt_data
            else tt_data["features"].shape[0]
        ),
        "context_len": args.context_len,
        "stride": args.stride,
        "gamma": args.gamma,
        "tt_discretize": bool(args.tt_discretize),
        "tt_num_bins": int(args.tt_num_bins),
        "output_format": args.output_format,
        "state_dim": int(episodes[0].states.shape[-1]),
        "action_dim": int(episodes[0].actions.shape[-1]),
    }
    (out_dir / f"{args.prefix}_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Conversion complete.")
    print(f"Scenarios processed: {metadata['n_scenarios']}")
    print(f"Episodes extracted: {metadata['n_episodes']}")
    print(f"DT samples: {metadata['dt_num_samples']}")
    print(f"TT samples: {metadata['tt_num_samples']}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
