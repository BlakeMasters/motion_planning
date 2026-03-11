from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

DEFAULT_TFRECORD = (
    "uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000"
)

OBJECT_TYPE_COLORS = {
    0: "tab:gray",
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    4: "tab:red",
}


@dataclass
class PredictionOverlay:
    pred_xy: np.ndarray
    true_xy: np.ndarray
    valid_mask: np.ndarray
    scenario_index: int | None
    scenario_id: str | None
    ade_m: float | None
    fde_m: float | None


def _iter_scenarios(path: str):
    from waymo_open_dataset.protos import scenario_pb2

    dataset = tf.data.TFRecordDataset(path, compression_type="")
    for i, raw_record in enumerate(dataset):  # pragma: no branch - linear scan
        yield i, scenario_pb2.Scenario.FromString(raw_record.numpy())


def load_scenario(path: str, scenario_index: int | None = None, scenario_id: str | None = None) -> tuple[Any, int]:
    if scenario_id is not None and scenario_id != "":
        for i, scenario in _iter_scenarios(path):
            if scenario.scenario_id == scenario_id:
                return scenario, i
        raise ValueError(f"Scenario id '{scenario_id}' not found in {path}.")

    if scenario_index is None:
        scenario_index = 0
    for i, scenario in _iter_scenarios(path):
        if i == scenario_index:
            return scenario, i
    raise IndexError(f"Scenario index {scenario_index} is out of bounds for {path}.")


def draw_track(ax: plt.Axes, track: Any) -> None:
    valids = np.array([state.valid for state in track.states], dtype=bool)
    if not np.any(valids):
        return

    x = np.array([state.center_x for state in track.states], dtype=float)
    y = np.array([state.center_y for state in track.states], dtype=float)
    color = OBJECT_TYPE_COLORS.get(track.object_type, "tab:purple")

    ax.plot(x[valids], y[valids], linewidth=1.5, alpha=0.9, color=color)

    last_valid_idx = np.where(valids)[0][-1]
    ax.scatter(
        [x[last_valid_idx]],
        [y[last_valid_idx]],
        s=8,
        color=color,
        alpha=0.9,
    )


def draw_prediction_overlay(ax: plt.Axes, overlay: PredictionOverlay, model_label: str = "Decision Transformer") -> None:
    valid = overlay.valid_mask.astype(bool)
    if valid.sum() == 0:
        return

    pred_xy = overlay.pred_xy
    true_xy = overlay.true_xy
    idx = np.where(valid)[0]

    ax.plot(
        true_xy[idx, 0],
        true_xy[idx, 1],
        linestyle="--",
        linewidth=2.0,
        color="black",
        label="Ground Truth (SDC)",
        zorder=20,
    )
    ax.plot(
        pred_xy[idx, 0],
        pred_xy[idx, 1],
        linestyle="-",
        linewidth=2.0,
        color="crimson",
        label=model_label,
        zorder=21,
    )
    ax.scatter(
        [true_xy[idx[0], 0]],
        [true_xy[idx[0], 1]],
        marker="o",
        s=40,
        color="gold",
        edgecolors="black",
        label="Prediction start",
        zorder=22,
    )
    ax.scatter(
        [true_xy[idx[-1], 0]],
        [true_xy[idx[-1], 1]],
        marker="x",
        s=40,
        color="black",
        label="GT end",
        zorder=22,
    )
    ax.scatter(
        [pred_xy[idx[-1], 0]],
        [pred_xy[idx[-1], 1]],
        marker="x",
        s=40,
        color="crimson",
        label="Pred end",
        zorder=22,
    )


def _get_sdc_track(scenario: Any):
    sdc_idx = int(getattr(scenario, "sdc_track_index", -1))
    if sdc_idx < 0 or sdc_idx >= len(scenario.tracks):
        return None
    return scenario.tracks[sdc_idx]


def _get_sdc_current_xy(scenario: Any) -> np.ndarray | None:
    track = _get_sdc_track(scenario)
    if track is None or len(track.states) == 0:
        return None
    states = track.states
    current_idx = min(10, len(states) - 1)
    if states[current_idx].valid:
        return np.array([states[current_idx].center_x, states[current_idx].center_y], dtype=np.float32)
    valid_idxs = [i for i, st in enumerate(states) if st.valid]
    if not valid_idxs:
        return np.array([states[current_idx].center_x, states[current_idx].center_y], dtype=np.float32)
    nearest = min(valid_idxs, key=lambda i: abs(i - current_idx))
    st = states[nearest]
    return np.array([st.center_x, st.center_y], dtype=np.float32)


def _get_sdc_future_xy(scenario: Any, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    track = _get_sdc_track(scenario)
    if track is None or horizon <= 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    states = track.states
    start_idx = 11  # 10 past + 1 current, then future begins.
    out_xy = np.zeros((horizon, 2), dtype=np.float32)
    out_valid = np.zeros((horizon,), dtype=bool)
    for t in range(horizon):
        idx = start_idx + t
        if idx >= len(states):
            break
        st = states[idx]
        out_xy[t, 0] = st.center_x
        out_xy[t, 1] = st.center_y
        out_valid[t] = bool(st.valid)
    return out_xy, out_valid


def _fit_rigid_2d(src_xy: np.ndarray, tgt_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = src_xy.mean(axis=0)
    tgt_mean = tgt_xy.mean(axis=0)
    x = src_xy - src_mean
    y = tgt_xy - tgt_mean
    u, _, vt = np.linalg.svd(x.T @ y)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    trans = tgt_mean - src_mean @ rot
    return rot, trans


def _apply_rigid_2d(xy: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return xy @ rot + trans


def maybe_align_overlay_to_sdc_future(scenario: Any, overlay: PredictionOverlay) -> tuple[bool, float, float]:
    horizon = int(overlay.true_xy.shape[0])
    if horizon == 0:
        return False, float("nan"), float("nan")

    sdc_future_xy, sdc_valid = _get_sdc_future_xy(scenario, horizon)
    if sdc_future_xy.shape[0] == 0:
        return False, float("nan"), float("nan")

    n = min(horizon, sdc_future_xy.shape[0], overlay.valid_mask.shape[0])
    pair_valid = overlay.valid_mask[:n].astype(bool) & sdc_valid[:n]
    if pair_valid.sum() == 0:
        return False, float("nan"), float("nan")

    src = overlay.true_xy[:n][pair_valid]
    tgt = sdc_future_xy[:n][pair_valid]
    before_rmse = float(np.sqrt(np.mean(np.sum((src - tgt) ** 2, axis=1))))

    if src.shape[0] >= 2:
        rot, trans = _fit_rigid_2d(src, tgt)
    else:
        rot = np.eye(2, dtype=np.float32)
        trans = (tgt[0] - src[0]).astype(np.float32)

    true_aligned = _apply_rigid_2d(overlay.true_xy, rot, trans)
    pred_aligned = _apply_rigid_2d(overlay.pred_xy, rot, trans)
    after_src = true_aligned[:n][pair_valid]
    after_rmse = float(np.sqrt(np.mean(np.sum((after_src - tgt) ** 2, axis=1))))

    # Apply only when alignment clearly improves mismatch.
    should_apply = np.isfinite(after_rmse) and (
        (before_rmse > 50.0 and after_rmse < 20.0)
        or (after_rmse < before_rmse * 0.5 and before_rmse > 5.0)
    )
    if should_apply:
        overlay.true_xy = true_aligned
        overlay.pred_xy = pred_aligned
        return True, before_rmse, after_rmse
    return False, before_rmse, after_rmse


def _apply_zoom(ax: plt.Axes, scenario: Any, overlay: PredictionOverlay | None, zoom_meters: float) -> None:
    if zoom_meters <= 0:
        return

    center = _get_sdc_current_xy(scenario)
    if center is None:
        center = np.array([0.0, 0.0], dtype=np.float32)

    if overlay is not None:
        valid = overlay.valid_mask.astype(bool)
        if valid.any():
            overlay_center = overlay.true_xy[valid].mean(axis=0)
            if np.linalg.norm(overlay_center - center) < 3.0 * zoom_meters:
                center = 0.5 * center + 0.5 * overlay_center

    ax.set_xlim(float(center[0] - zoom_meters), float(center[0] + zoom_meters))
    ax.set_ylim(float(center[1] - zoom_meters), float(center[1] + zoom_meters))


def load_prediction_overlay(path: Path, prediction_index: int) -> PredictionOverlay:
    payload = np.load(path)
    pred_xy = payload["pred_xy"][prediction_index]
    true_xy = payload["true_xy"][prediction_index]
    valid_mask = payload["valid_mask"][prediction_index]
    scenario_index = None
    scenario_id = None
    ade_m = None
    fde_m = None

    if "scenario_index" in payload:
        scenario_index = int(payload["scenario_index"][prediction_index])
    if "scenario_id" in payload:
        scenario_id_raw = payload["scenario_id"][prediction_index]
        if isinstance(scenario_id_raw, np.ndarray) and scenario_id_raw.shape == ():
            scenario_id_raw = scenario_id_raw.item()
        if isinstance(scenario_id_raw, (bytes, np.bytes_)):
            scenario_id = scenario_id_raw.decode("utf-8", errors="ignore")
        else:
            scenario_id = str(scenario_id_raw)
    if "ade_m" in payload:
        ade_m = float(payload["ade_m"][prediction_index])
    if "fde_m" in payload:
        fde_m = float(payload["fde_m"][prediction_index])

    return PredictionOverlay(
        pred_xy=pred_xy,
        true_xy=true_xy,
        valid_mask=valid_mask,
        scenario_index=scenario_index,
        scenario_id=scenario_id,
        ade_m=ade_m,
        fde_m=fde_m,
    )


def visualize_scenario(
    scenario: Any,
    output_path: str,
    max_tracks: int | None = None,
    overlay: PredictionOverlay | None = None,
    zoom_meters: float = 120.0,
    model_label: str = "Decision Transformer",
) -> None:
    try:
        from waymo_open_dataset.utils.sim_agents import visualizations
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "waymo_open_dataset is required for visualization. "
            "Install waymo-open-dataset-tf-2-12-0 first."
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 10))
    visualizations.add_map(ax, scenario)

    tracks = scenario.tracks if max_tracks is None else scenario.tracks[:max_tracks]
    for track in tracks:
        draw_track(ax, track)

    if overlay is not None:
        draw_prediction_overlay(ax, overlay, model_label=model_label)

    ax.set_aspect("equal", adjustable="box")
    title = (
        f"Scenario {scenario.scenario_id}\n"
        f"tracks={len(tracks)}/{len(scenario.tracks)}  "
        f"map_features={len(scenario.map_features)}"
    )
    if overlay is not None and overlay.ade_m is not None and overlay.fde_m is not None:
        title += f"\n{model_label} ADE={overlay.ade_m:.2f}m  FDE={overlay.fde_m:.2f}m"
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)
    _apply_zoom(ax, scenario, overlay, zoom_meters=zoom_meters)

    if overlay is not None:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one Waymo scenario and optional DT predictions."
    )
    parser.add_argument(
        "--tfrecord",
        type=str,
        default=DEFAULT_TFRECORD,
        help="Path to a Waymo Scenario TFRecord shard.",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=None,
        help="0-based scenario index in the TFRecord shard. If unset with --predictions, uses that bundle index.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="waymo_scenario.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="If set, visualize only the first N tracks.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Optional .npz file produced by train_decision_transformer_gcs.py (--output-test-predictions).",
    )
    parser.add_argument(
        "--prediction-index",
        type=int,
        default=0,
        help="Row index inside --predictions to visualize.",
    )
    parser.add_argument(
        "--zoom-meters",
        type=float,
        default=120.0,
        help="Half-width of local view window in meters around SDC/overlay (<=0 disables zoom).",
    )
    parser.add_argument(
        "--no-overlay-align",
        action="store_true",
        help="Disable automatic rigid alignment of overlay to scenario SDC future path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in a GUI window in addition to saving.",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default="Decision Transformer",
        help="Model name shown in the legend and title (e.g. 'Trajectory Transformer').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    overlay = None
    if args.predictions is not None:
        prediction_path = Path(args.predictions)
        if not prediction_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {prediction_path}")
        overlay = load_prediction_overlay(prediction_path, args.prediction_index)

    scenario_index = args.scenario_index

    tfrecord_path = Path(args.tfrecord)
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")

    resolved_index = scenario_index
    if overlay is not None and overlay.scenario_id:
        scenario, resolved_index = load_scenario(
            str(tfrecord_path),
            scenario_index=scenario_index,
            scenario_id=overlay.scenario_id,
        )
        if scenario_index is not None and resolved_index != scenario_index:
            raise ValueError(
                f"Prediction row maps to scenario_id={overlay.scenario_id} at index={resolved_index}, "
                f"but --scenario-index {scenario_index} was requested."
            )
    else:
        if resolved_index is None:
            if overlay is not None and overlay.scenario_index is not None and overlay.scenario_index >= 0:
                resolved_index = overlay.scenario_index
            else:
                resolved_index = 0
        scenario, resolved_index = load_scenario(str(tfrecord_path), scenario_index=resolved_index)

    if overlay is not None and overlay.scenario_id and scenario.scenario_id != overlay.scenario_id:
        raise ValueError(
            f"Scenario mismatch: prediction scenario_id={overlay.scenario_id}, "
            f"loaded scenario_id={scenario.scenario_id}."
        )

    if overlay is not None and not args.no_overlay_align:
        aligned, rmse_before, rmse_after = maybe_align_overlay_to_sdc_future(scenario, overlay)
        if aligned:
            print(
                f"Applied overlay rigid alignment to SDC future path "
                f"(rmse_before={rmse_before:.2f}m -> rmse_after={rmse_after:.2f}m)"
            )
        else:
            if np.isfinite(rmse_before) and np.isfinite(rmse_after):
                print(
                    f"Overlay alignment not applied "
                    f"(rmse_before={rmse_before:.2f}m, rmse_after={rmse_after:.2f}m)"
                )

    visualize_scenario(
        scenario,
        args.output,
        args.max_tracks,
        overlay=overlay,
        zoom_meters=args.zoom_meters,
        model_label=args.model_label,
    )
    print(f"Saved visualization to: {args.output}")
    print(f"Scenario index: {resolved_index}")
    if overlay is not None and overlay.scenario_id:
        print(f"Scenario id: {overlay.scenario_id}")
    if overlay is not None and overlay.ade_m is not None and overlay.fde_m is not None:
        print(f"ADE: {overlay.ade_m:.3f} m | FDE: {overlay.fde_m:.3f} m")

    if args.show:
        image = plt.imread(args.output)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
