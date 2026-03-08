import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils.sim_agents import visualizations


DEFAULT_TFRECORD = (
    "uncompressed_scenario_training_20s_training_20s.tfrecord-00000-of-01000"
)

OBJECT_TYPE_COLORS = {
    scenario_pb2.Track.ObjectType.TYPE_UNSET: "tab:gray",
    scenario_pb2.Track.ObjectType.TYPE_VEHICLE: "tab:blue",
    scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN: "tab:orange",
    scenario_pb2.Track.ObjectType.TYPE_CYCLIST: "tab:green",
    scenario_pb2.Track.ObjectType.TYPE_OTHER: "tab:red",
}


def load_scenario(path: str, scenario_index: int) -> scenario_pb2.Scenario:
    dataset = tf.data.TFRecordDataset(path, compression_type="")
    for i, raw_record in enumerate(dataset):
        if i == scenario_index:
            return scenario_pb2.Scenario.FromString(raw_record.numpy())
    raise IndexError(f"Scenario index {scenario_index} is out of bounds for {path}.")


def draw_track(ax: plt.Axes, track: scenario_pb2.Track) -> None:
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


def visualize_scenario(
    scenario: scenario_pb2.Scenario,
    output_path: str,
    max_tracks: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    visualizations.add_map(ax, scenario)

    tracks = scenario.tracks if max_tracks is None else scenario.tracks[:max_tracks]
    for track in tracks:
        draw_track(ax, track)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Scenario {scenario.scenario_id}\n"
        f"tracks={len(tracks)}/{len(scenario.tracks)}  "
        f"map_features={len(scenario.map_features)}"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one Waymo scenario from a TFRecord file."
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
        default=0,
        help="0-based scenario index in the TFRecord shard.",
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
        "--show",
        action="store_true",
        help="Display the figure in a GUI window in addition to saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    tfrecord_path = Path(args.tfrecord)
    if not tfrecord_path.exists():
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")

    scenario = load_scenario(str(tfrecord_path), args.scenario_index)
    visualize_scenario(scenario, args.output, args.max_tracks)
    print(f"Saved visualization to: {args.output}")

    if args.show:
        image = plt.imread(args.output)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
