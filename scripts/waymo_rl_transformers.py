# Install dependencies (run once; skip if already installed)
# The Docker images (Dockerfile / Dockerfile.gpu) handle this automatically.
import subprocess, sys

packages = [
    "waymo-open-dataset-tf-2-12-0==1.6.7",
    "torch==2.4.1",
    "torchvision==0.19.1",
    "transformers>=4.46,<5.0",   # 4.46+ requires PyTorch >= 2.4
    "einops",
    "google-cloud-storage",
    "pandas",
    "seaborn",
    "scikit-learn",
    "ipywidgets",
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("All packages ready.")


# ── Standard library ──────────────────────────────────────────────────────────
import os, math, json, uuid, time, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

# ── Matplotlib ────────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from IPython.display import HTML, display
warnings.filterwarnings("ignore")

# ── TensorFlow (data loading) ─────────────────────────────────────────────────
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ── PyTorch (models) ──────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model

# ── Sklearn (evaluation helpers) ─────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"TensorFlow {tf.__version__}  |  PyTorch {torch.__version__}  |  device={DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION  (edit these to match your environment)
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # ── Data ──────────────────────────────────────────────────────────────────
    GCS_BUCKET     = "waymo_open_dataset_motion_v_1_2_0",
    GCS_TRAIN_PATH = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/training",
    GCS_VAL_PATH   = "gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/tf_example/validation",
    NUM_TRAIN_SHARDS = 8,          # Use 8/1000 shards for quick iteration
    NUM_VAL_SHARDS   = 2,
    LOCAL_DATA_DIR   = "./data_cache",  # TFRecords cached here after first load

    # ── Scene / agents ────────────────────────────────────────────────────────
    N_AGENTS       = 128,
    N_PAST         = 10,
    N_CURRENT      = 1,
    N_FUTURE       = 80,
    N_MAP_SAMPLES  = 30000,
    DT_SEC         = 0.1,           # 10 Hz

    # ── RL / sequence ─────────────────────────────────────────────────────────
    STATE_DIM      = 16,            # ego(6) + 5 nearest agents × 2 dims
    ACT_DIM        = 2,             # (Δx, Δy) in AV-local frame
    CONTEXT_LEN    = 20,            # transformer look-back window
    PRED_HORIZON   = 16,            # future steps to predict (every 5 frames = 0.5s each)
    RTG_SCALE      = 10.0,          # normalisation for return-to-go

    # ── DT model ─────────────────────────────────────────────────────────────
    DT_HIDDEN      = 256 if torch.cuda.is_available() else 128,
    DT_N_LAYER     = 4   if torch.cuda.is_available() else 3,
    DT_N_HEAD      = 4   if torch.cuda.is_available() else 1,
    DT_DROPOUT     = 0.1,
    DT_LR          = 1e-4,
    DT_EPOCHS      = 20,
    DT_BATCH       = 64  if torch.cuda.is_available() else 16,

    # ── TT model ─────────────────────────────────────────────────────────────
    TT_HIDDEN      = 256 if torch.cuda.is_available() else 128,
    TT_N_LAYER     = 4   if torch.cuda.is_available() else 3,
    TT_N_HEAD      = 4   if torch.cuda.is_available() else 1,
    TT_N_BINS      = 64,
    TT_DROPOUT     = 0.1,
    TT_LR          = 1e-4,
    TT_EPOCHS      = 20,
    TT_BATCH       = 32  if torch.cuda.is_available() else 8,
    TT_BEAM_SIZE   = 4,

    # ── Reproducibility ───────────────────────────────────────────────────────
    SEED           = 42,
)

torch.manual_seed(CFG["SEED"])
np.random.seed(CFG["SEED"])
print("Configuration loaded.")


import subprocess, os

def check_gcs_access(bucket: str) -> bool:
    """Return True if GCS is reachable with current credentials."""
    try:
        from google.cloud import storage
        project = (os.environ.get("GOOGLE_CLOUD_PROJECT") or
                   os.environ.get("GCLOUD_PROJECT") or
                   os.environ.get("GCP_PROJECT"))
        client = storage.Client(project=project) if project else storage.Client()
        b = client.bucket(bucket)
        next(client.list_blobs(b, max_results=1))
        return True
    except Exception as e:
        print(f"GCS not accessible: {e}")
        return False

GCS_OK = check_gcs_access(CFG["GCS_BUCKET"])

if GCS_OK:
    print("✓ GCS credentials valid — will stream TFRecords from GCS")
else:
    print("✗ GCS not configured — synthetic data will be used for model demos")
    print("  Follow the instructions in Section 2 to authenticate.")


NUM_MAP = CFG["N_MAP_SAMPLES"]

roadgraph_features = {
    "roadgraph_samples/dir":   tf.io.FixedLenFeature([NUM_MAP, 3], tf.float32),
    "roadgraph_samples/id":    tf.io.FixedLenFeature([NUM_MAP, 1], tf.int64),
    "roadgraph_samples/type":  tf.io.FixedLenFeature([NUM_MAP, 1], tf.int64),
    "roadgraph_samples/valid": tf.io.FixedLenFeature([NUM_MAP, 1], tf.int64),
    "roadgraph_samples/xyz":   tf.io.FixedLenFeature([NUM_MAP, 3], tf.float32),
}

state_features = {}
for split, T in [("past", CFG["N_PAST"]), ("current", CFG["N_CURRENT"]), ("future", CFG["N_FUTURE"])]:
    for feat in ["x", "y", "z", "bbox_yaw", "vel_yaw", "velocity_x", "velocity_y",
                 "length", "width", "height", "valid", "timestamp_micros"]:
        dtype = tf.int64 if feat in ("valid", "timestamp_micros") else tf.float32
        state_features[f"state/{split}/{feat}"] = tf.io.FixedLenFeature(
            [CFG["N_AGENTS"], T], dtype)

for key in ("id", "type", "is_sdc", "tracks_to_predict"):
    dtype = tf.float32 if key in ("id", "type") else tf.int64
    state_features[f"state/{key}"] = tf.io.FixedLenFeature([CFG["N_AGENTS"]], dtype)

traffic_light_features = {}
for split, T in [("past", CFG["N_PAST"]), ("current", 1)]:
    for feat in ["state", "valid", "x", "y", "z"]:
        dtype = tf.int64 if feat in ("state", "valid") else tf.float32
        traffic_light_features[f"traffic_light_state/{split}/{feat}"] =             tf.io.FixedLenFeature([T, 16], dtype)

FEATURES = {}
FEATURES.update(roadgraph_features)
FEATURES.update(state_features)
FEATURES.update(traffic_light_features)

print(f"Feature schema: {len(FEATURES)} keys defined")


import glob as _glob

def list_shards(path: str, n: int) -> list:
    """Return up to n shard paths from a GCS prefix or local directory."""
    if path.startswith("gs://"):
        from google.cloud import storage
        client  = storage.Client()
        # strip gs://bucket/
        parts   = path[5:].split("/", 1)
        bucket  = parts[0]
        prefix  = parts[1] if len(parts) > 1 else ""
        blobs   = list(client.list_blobs(bucket, prefix=prefix, max_results=n * 2))
        paths   = [f"gs://{bucket}/{b.name}" for b in blobs
                   if b.name.endswith(".tfrecord") or "tfrecord" in b.name]
        return paths[:n]
    else:
        paths = sorted(_glob.glob(os.path.join(path, "*.tfrecord*")))
        return paths[:n]


def parse_scenario(raw: bytes) -> dict:
    """Parse one tf.Example and return numpy arrays."""
    ex = tf.io.parse_single_example(raw, FEATURES)

    past   = tf.stack([ex["state/past/x"],   ex["state/past/y"],
                       ex["state/past/velocity_x"], ex["state/past/velocity_y"],
                       tf.math.cos(ex["state/past/bbox_yaw"]),
                       tf.math.sin(ex["state/past/bbox_yaw"])], axis=-1)     # [128, 10, 6]
    cur    = tf.stack([ex["state/current/x"],   ex["state/current/y"],
                       ex["state/current/velocity_x"], ex["state/current/velocity_y"],
                       tf.math.cos(ex["state/current/bbox_yaw"]),
                       tf.math.sin(ex["state/current/bbox_yaw"])], axis=-1)  # [128, 1, 6]
    future = tf.stack([ex["state/future/x"],   ex["state/future/y"],
                       ex["state/future/velocity_x"], ex["state/future/velocity_y"],
                       tf.math.cos(ex["state/future/bbox_yaw"]),
                       tf.math.sin(ex["state/future/bbox_yaw"])], axis=-1)   # [128, 80, 6]

    hist_xy = tf.concat([past[..., :2], cur[..., :2]], axis=1)  # [128, 11, 2]

    return {
        "history":    tf.concat([past, cur], axis=1),            # [128, 11, 6]
        "future":     future,                                     # [128, 80, 6]
        "hist_xy":    hist_xy,                                    # [128, 11, 2]
        "fut_valid":  tf.concat([ex["state/past/valid"],
                                 ex["state/current/valid"],
                                 ex["state/future/valid"]], axis=1),  # [128, 91]
        "is_sdc":     ex["state/is_sdc"],                        # [128]
        "type":       ex["state/type"],                          # [128]
        "tracks":     ex["state/tracks_to_predict"],             # [128]
        "rg_xyz":     ex["roadgraph_samples/xyz"],               # [30000, 3]
        "rg_type":    ex["roadgraph_samples/type"],              # [30000, 1]
        "rg_valid":   ex["roadgraph_samples/valid"],             # [30000, 1]
    }


def build_dataset(path: str, n_shards: int, batch: int = 1,
                  repeat: bool = False) -> tf.data.Dataset:
    shards = list_shards(path, n_shards)
    if not shards:
        print(f"  No shards found at {path} — returning None")
        return None
    print(f"  Found {len(shards)} shards; first: {shards[0]}")
    ds = (tf.data.TFRecordDataset(shards, num_parallel_reads=4)
            .map(parse_scenario, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch)
            .prefetch(tf.data.AUTOTUNE))
    if repeat:
        ds = ds.repeat()
    return ds


if GCS_OK:
    print("Building training dataset from GCS...")
    train_ds = build_dataset(CFG["GCS_TRAIN_PATH"], CFG["NUM_TRAIN_SHARDS"], batch=1)
    print("Building validation dataset from GCS...")
    val_ds   = build_dataset(CFG["GCS_VAL_PATH"],   CFG["NUM_VAL_SHARDS"],   batch=1)
else:
    train_ds = val_ds = None
    print("Skipping real dataset — synthetic data used in model cells below.")


# ── Road-graph type colours ────────────────────────────────────────────────────
LANE_TYPES       = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16}
ROAD_EDGE_TYPES  = {15, 16}
STOP_SIGN_TYPES  = {17}

TYPE_COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#9467bd"}   # vehicle, ped, cyclist
AV_COLOR    = "#ff7f0e"

def render_scene(ax, rg_xyz, rg_type, rg_valid,
                 history, future, is_sdc, agent_type,
                 fut_valid, step: int = 10,
                 pred_future=None, title: str = ""):
    """
    Draw one scene frame on ax.
    history:  [N_agents, T_hist, 2]   (x, y)
    future:   [N_agents, T_fut,  2]   (x, y)
    step:     which future step to highlight (0-79)
    pred_future: optional [N_agents, T_fut, 2] model predictions
    """
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")

    # ── Road graph ─────────────────────────────────────────────────────────────
    valid_mask = rg_valid[:, 0].astype(bool)
    types      = rg_type[valid_mask, 0]
    xyz        = rg_xyz[valid_mask]

    lane_mask  = np.isin(types, list(LANE_TYPES))
    edge_mask  = np.isin(types, list(ROAD_EDGE_TYPES))
    stop_mask  = np.isin(types, list(STOP_SIGN_TYPES))

    ax.scatter(xyz[lane_mask, 0], xyz[lane_mask, 1], s=0.3, c="#555577", zorder=1)
    ax.scatter(xyz[edge_mask, 0], xyz[edge_mask, 1], s=0.5, c="#aaaacc", zorder=2)
    if stop_mask.any():
        ax.scatter(xyz[stop_mask, 0], xyz[stop_mask, 1],
                   s=30, c="#d62728", marker="x", zorder=5, label="Stop sign")

    # ── Agent trajectories ────────────────────────────────────────────────────
    for i in range(history.shape[0]):
        hist_valid = history[i, :, 0] != 0
        if not hist_valid.any():
            continue
        color = AV_COLOR if is_sdc[i] else TYPE_COLORS.get(int(agent_type[i]), "#888888")
        lw = 2.0 if is_sdc[i] else 0.8
        zorder = 8 if is_sdc[i] else 4

        # History
        ax.plot(history[i, hist_valid, 0], history[i, hist_valid, 1],
                color=color, lw=lw, alpha=0.9, zorder=zorder)

        # Ground-truth future (dashed)
        fut_v = fut_valid[i, :future.shape[1]].astype(bool)
        if fut_v.any():
            ax.plot(future[i, fut_v, 0], future[i, fut_v, 1],
                    color=color, lw=lw, ls="--", alpha=0.4, zorder=zorder)

        # Predicted future (solid bright line)
        if pred_future is not None:
            ax.plot(pred_future[i, fut_v, 0], pred_future[i, fut_v, 1],
                    color="yellow", lw=1.2, ls="-", alpha=0.8, zorder=zorder + 1)

        # Current position dot
        cx, cy = history[i, -1, 0], history[i, -1, 1]
        ax.scatter([cx], [cy], s=30 if is_sdc[i] else 10,
                   c=color, zorder=zorder + 2, edgecolors="white", linewidths=0.3)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=AV_COLOR,   label="AV (SDC)"),
        mpatches.Patch(color="#1f77b4",  label="Vehicle"),
        mpatches.Patch(color="#2ca02c",  label="Pedestrian"),
        mpatches.Patch(color="#9467bd",  label="Cyclist"),
        Line2D([0],[0], color="white", ls="--", alpha=0.5, label="GT future"),
    ]
    if pred_future is not None:
        handles.append(Line2D([0],[0], color="yellow", ls="-", label="Predicted"))
    ax.legend(handles=handles, loc="upper left", fontsize=6,
              facecolor="#1a1a2e", labelcolor="white", framealpha=0.7)
    ax.set_title(title, color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")


def plot_scenario(scenario: dict, title: str = ""):
    """Convenience: render a single parsed scenario dict."""
    # Strip batch dim if present (handles any rank with batch_size=1)
    def _squeeze(x):
        return x[0] if x.ndim >= 2 and x.shape[0] == 1 else x

    hist    = _squeeze(scenario["history"].numpy())[..., :2]    # [128, 11, 2]
    fut     = _squeeze(scenario["future"].numpy())[..., :2]     # [128, 80, 2]
    fv      = _squeeze(scenario["fut_valid"].numpy())[:, 11:]   # [128, 80]
    sdc     = _squeeze(scenario["is_sdc"].numpy())
    atype   = _squeeze(scenario["type"].numpy())
    rg_xyz  = _squeeze(scenario["rg_xyz"].numpy())
    rg_type = _squeeze(scenario["rg_type"].numpy())
    rg_val  = _squeeze(scenario["rg_valid"].numpy())

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#1a1a2e")
    render_scene(ax, rg_xyz, rg_type, rg_val,
                 hist, fut, sdc, atype, fv, title=title)
    plt.tight_layout()
    plt.show()


def create_animation(frames_list: list, interval: int = 100) -> animation.FuncAnimation:
    """Create an animation from a list of rendered RGBA frames."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames_list[0])
    ax.axis("off")
    def _upd(i):
        im.set_data(frames_list[i])
        return (im,)
    anim = animation.FuncAnimation(fig, _upd, frames=len(frames_list),
                                   interval=interval, blit=True)
    plt.close(fig)
    return anim


# ── Quick demo ────────────────────────────────────────────────────────────────
if train_ds is not None:
    sample = next(iter(train_ds))
    print("Rendering first training scenario...")
    plot_scenario(sample, title="Training Scenario — Ground Truth Trajectories")
else:
    print("No real data available — visualisation skipped (run after GCS auth).")


def compute_eda_stats(ds, max_scenarios: int = 200) -> pd.DataFrame:
    """Collect per-scenario stats into a DataFrame."""
    rows = []
    for i, sc in enumerate(ds):
        if i >= max_scenarios:
            break
        atype = sc["type"][0].numpy()          # [128]
        sdc   = sc["is_sdc"][0].numpy()        # [128]
        valid = sc["fut_valid"][0].numpy()      # [128, 91]
        hist  = sc["history"][0].numpy()        # [128, 11, 6]  6=(x,y,vx,vy,cosH,sinH)

        n_vehicles = int((atype == 1).sum())
        n_peds     = int((atype == 2).sum())
        n_cyclists = int((atype == 3).sum())

        sdc_idx = np.where(sdc)[0]
        sdc_speed = 0.0
        if len(sdc_idx):
            vx, vy = hist[sdc_idx[0], -1, 2], hist[sdc_idx[0], -1, 3]
            sdc_speed = float(np.sqrt(vx**2 + vy**2))

        rows.append({
            "scenario": i,
            "n_vehicles": n_vehicles,
            "n_pedestrians": n_peds,
            "n_cyclists": n_cyclists,
            "sdc_speed_ms": sdc_speed,
            "n_valid_agents": int((valid[:, -1] > 0).sum()),
        })

    return pd.DataFrame(rows)


if train_ds is not None:
    print("Computing EDA stats (first 200 scenarios)...")
    eda_df = compute_eda_stats(train_ds)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Waymo Open Motion Dataset — EDA", fontsize=14)

    axes[0,0].hist(eda_df["n_vehicles"],    bins=20, color="#1f77b4", edgecolor="white")
    axes[0,0].set_title("Vehicles per Scene")
    axes[0,0].set_xlabel("Count"); axes[0,0].set_ylabel("Scenarios")

    axes[0,1].hist(eda_df["n_pedestrians"], bins=20, color="#2ca02c", edgecolor="white")
    axes[0,1].set_title("Pedestrians per Scene")

    axes[0,2].hist(eda_df["n_cyclists"],    bins=20, color="#9467bd", edgecolor="white")
    axes[0,2].set_title("Cyclists per Scene")

    axes[1,0].hist(eda_df["sdc_speed_ms"],  bins=30, color="#ff7f0e", edgecolor="white")
    axes[1,0].set_title("AV Speed at t=0 (m/s)"); axes[1,0].set_xlabel("Speed (m/s)")

    axes[1,1].hist(eda_df["n_valid_agents"], bins=20, color="#d62728", edgecolor="white")
    axes[1,1].set_title("Valid Agents at Final Timestep")

    agent_means = eda_df[["n_vehicles","n_pedestrians","n_cyclists"]].mean()
    axes[1,2].bar(["Vehicles","Pedestrians","Cyclists"], agent_means,
                  color=["#1f77b4","#2ca02c","#9467bd"])
    axes[1,2].set_title("Mean Agent Count per Scene")

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("eda_stats.png", dpi=120)
    plt.show()
    print(eda_df.describe().round(2))
else:
    # Synthetic EDA for demo
    print("Generating synthetic EDA demo...")
    np.random.seed(42)
    n = 200
    synth = pd.DataFrame({
        "n_vehicles":    np.random.poisson(20, n),
        "n_pedestrians": np.random.poisson(4, n),
        "n_cyclists":    np.random.poisson(2, n),
        "sdc_speed_ms":  np.abs(np.random.normal(8, 4, n)),
        "n_valid_agents":np.random.randint(5, 40, n),
    })
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("EDA Demo (Synthetic Data — replace with real WOMD)", fontsize=11)
    synth["n_vehicles"].hist(ax=axes[0], bins=20, color="#1f77b4")
    axes[0].set_title("Vehicles per Scene")
    synth["sdc_speed_ms"].hist(ax=axes[1], bins=30, color="#ff7f0e")
    axes[1].set_title("AV Speed (m/s)")
    synth["n_valid_agents"].hist(ax=axes[2], bins=20, color="#d62728")
    axes[2].set_title("Valid Agents at Final Step")
    plt.tight_layout(); plt.show()


# ── Coordinate helpers ────────────────────────────────────────────────────────
POS_SCALE  = 50.0    # metres
VEL_SCALE  = 15.0    # m/s
ACT_SCALE  = 1.5     # max displacement per step (m)
K_NEAREST  = 5       # nearest neighbours included in state

def to_av_local(x, y, cx, cy, cos_h, sin_h):
    """Rotate world (x,y) to AV-centric frame."""
    dx = x - cx
    dy = y - cy
    xl = dx * cos_h + dy * sin_h
    yl = -dx * sin_h + dy * cos_h
    return xl, yl


def build_state(hist: np.ndarray, sdc_idx: int, t: int) -> np.ndarray:
    """
    Build 16-dim state vector at time t.
    hist: [N_agents, T_hist, 6]  (x, y, vx, vy, cosH, sinH)
    t:    index into T_hist (typically the last frame = 10)
    """
    ego = hist[sdc_idx, t]            # (x, y, vx, vy, cosH, sinH)
    ex, ey, evx, evy, cosh, sinh = ego

    state = np.zeros(CFG["STATE_DIM"], dtype=np.float32)
    # Ego components (6 dims)
    state[0] = ex  / POS_SCALE
    state[1] = ey  / POS_SCALE
    state[2] = cosh                   # already unit-norm
    state[3] = sinh
    state[4] = evx / VEL_SCALE
    state[5] = evy / VEL_SCALE

    # K nearest agents (2 dims each = 10 dims)
    others = []
    for i in range(hist.shape[0]):
        if i == sdc_idx:
            continue
        ox, oy = hist[i, t, 0], hist[i, t, 1]
        if ox == 0 and oy == 0:
            continue
        dist = math.sqrt((ox - ex)**2 + (oy - ey)**2)
        others.append((dist, i))
    others.sort()

    slot = 6
    for _, idx in others[:K_NEAREST]:
        dx = (hist[idx, t, 0] - ex) / POS_SCALE
        dy = (hist[idx, t, 1] - ey) / POS_SCALE
        state[slot]   = dx
        state[slot+1] = dy
        slot += 2

    return state


def build_action(fut: np.ndarray, sdc_idx: int, t: int,
                 hist: np.ndarray) -> np.ndarray:
    """(Δx_local, Δy_local) displacement rotated into AV-local heading frame."""
    if t == 0:
        return np.zeros(2, dtype=np.float32)
    dx = (fut[sdc_idx, t, 0] - fut[sdc_idx, t-1, 0]) / ACT_SCALE
    dy = (fut[sdc_idx, t, 1] - fut[sdc_idx, t-1, 1]) / ACT_SCALE
    # Rotate world-frame displacement into AV-local frame using heading at t=10
    cos_h = hist[sdc_idx, 10, 4]
    sin_h = hist[sdc_idx, 10, 5]
    xl =  dx * cos_h + dy * sin_h
    yl = -dx * sin_h + dy * cos_h
    return np.array([xl, yl], dtype=np.float32)


def compute_rtg(fut: np.ndarray, sdc_idx: int, horizon: int) -> np.ndarray:
    """
    Return-to-go as cumulative negative step displacement, clipped to [-1, 0].
    r[t] = -||Δpos[t]||  (negative path length: less movement = higher reward).
    RTG[t] = Σ_{k>=t} r[k] / RTG_SCALE, clipped to [-1, 0].
    At inference desired_rtg=-0.8 represents typical human-speed driving.
    Shape: [horizon]
    """
    rewards = np.zeros(horizon, dtype=np.float32)
    for t in range(1, horizon):
        dx = fut[sdc_idx, t, 0] - fut[sdc_idx, t-1, 0]
        dy = fut[sdc_idx, t, 1] - fut[sdc_idx, t-1, 1]
        rewards[t] = -math.sqrt(dx**2 + dy**2)   # ≤ 0

    rtg = np.zeros(horizon, dtype=np.float32)
    running = 0.0
    for t in range(horizon - 1, -1, -1):
        running += rewards[t]
        rtg[t] = running / CFG["RTG_SCALE"]

    return np.clip(rtg, -1.0, 0.0)


print("Feature engineering utilities ready.")


class WOMDOfflineRLDataset(Dataset):
    """
    Offline RL dataset built from WOMD TFRecords.
    Iterates through parsed scenarios, extracts the SDC trajectory,
    and returns sliding windows of length `context_len`.
    """

    def __init__(self, tf_dataset, max_scenarios: int = 500,
                 context_len: int = None, pred_horizon: int = None):
        super().__init__()
        self.context_len  = context_len  or CFG["CONTEXT_LEN"]
        self.pred_horizon = pred_horizon or CFG["PRED_HORIZON"]
        self.state_dim    = CFG["STATE_DIM"]
        self.act_dim      = CFG["ACT_DIM"]
        self.trajectories = []           # list of dicts

        if tf_dataset is None:
            self._make_synthetic(max_scenarios)
        else:
            self._load_from_tf(tf_dataset, max_scenarios)

    def _load_from_tf(self, ds, max_sc: int):
        print(f"Extracting trajectories from up to {max_sc} scenarios...")
        for i, sc in enumerate(ds):
            if i >= max_sc:
                break
            hist = sc["history"][0].numpy()   # [128, 11, 6]
            fut  = sc["future"][0].numpy()    # [128, 80, 6]
            sdc  = sc["is_sdc"][0].numpy()    # [128]
            fv   = sc["fut_valid"][0].numpy() # [128, 91]

            sdc_idxs = np.where(sdc > 0)[0]
            if len(sdc_idxs) == 0:
                continue
            sdc_idx = int(sdc_idxs[0])

            T = self.pred_horizon
            states  = np.array([build_state(hist, sdc_idx, 10) for _ in range(T)])
            actions = np.array([build_action(fut, sdc_idx, t, hist)  for t in range(T)])
            rtg     = compute_rtg(fut, sdc_idx, T)

            self.trajectories.append({
                "states":  states,          # [T, state_dim]
                "actions": actions,         # [T, act_dim]
                "rtg":     rtg,             # [T]
                "timesteps": np.arange(T),  # [T]
            })
        print(f"  Loaded {len(self.trajectories)} trajectories.")

    def _make_synthetic(self, n: int):
        """Synthetic circular trajectories for architecture smoke-testing."""
        print(f"Creating {n} synthetic trajectories (no GCS data)...")
        T = self.pred_horizon
        for i in range(n):
            radius  = np.random.uniform(5, 30)
            omega   = np.random.uniform(0.05, 0.2)
            phase   = np.random.uniform(0, 2 * math.pi)

            states  = np.zeros((T, self.state_dim), dtype=np.float32)
            actions = np.zeros((T, self.act_dim),   dtype=np.float32)
            rtg     = np.zeros(T,                   dtype=np.float32)

            for t in range(T):
                angle = phase + omega * t
                x, y  = radius * math.cos(angle), radius * math.sin(angle)
                dx = -radius * omega * math.sin(angle) * 0.1
                dy =  radius * omega * math.cos(angle) * 0.1
                states[t, 0] = x / POS_SCALE
                states[t, 1] = y / POS_SCALE
                actions[t]   = np.array([dx, dy]) / ACT_SCALE
                step_len = math.sqrt(dx**2 + dy**2)
                rtg[t]   = max(-1.0, -(T - t) * step_len / CFG["RTG_SCALE"])

            self.trajectories.append({
                "states":    states.astype(np.float32),
                "actions":   actions.astype(np.float32),
                "rtg":       rtg.astype(np.float32),
                "timesteps": np.arange(T, dtype=np.int64),
            })

    def __len__(self) -> int:
        # Each trajectory produces multiple sliding windows
        return len(self.trajectories) * max(1, self.pred_horizon - self.context_len + 1)

    def __getitem__(self, idx: int) -> dict:
        traj_idx  = idx // max(1, self.pred_horizon - self.context_len + 1)
        start     = idx  % max(1, self.pred_horizon - self.context_len + 1)
        traj      = self.trajectories[traj_idx % len(self.trajectories)]
        T         = self.pred_horizon
        CL        = self.context_len

        # Extract window [start : start+CL] with zero-padding if needed
        def _window(arr):
            end = start + CL
            if end <= T:
                return arr[start:end].copy()
            pad = end - T
            return np.concatenate([arr[start:T], np.zeros((pad,) + arr.shape[1:],
                                   dtype=arr.dtype)], axis=0)

        states    = _window(traj["states"])                            # [CL, sd]
        actions   = _window(traj["actions"])                           # [CL, ad]
        rtg       = _window(traj["rtg"][:, None])                      # [CL, 1]
        timesteps = traj["timesteps"][start: start + CL]
        if len(timesteps) < CL:
            timesteps = np.concatenate([timesteps,
                np.full(CL - len(timesteps), timesteps[-1]+1, dtype=np.int64)])

        # Attention mask: 1 = real token, 0 = padding
        real      = min(CL, T - start)
        mask      = np.array([1]*real + [0]*(CL - real), dtype=np.int64)

        return {
            "states":         torch.tensor(states,    dtype=torch.float32),
            "actions":        torch.tensor(actions,   dtype=torch.float32),
            "returns_to_go":  torch.tensor(rtg,       dtype=torch.float32),
            "timesteps":      torch.tensor(timesteps, dtype=torch.long),
            "attention_mask": torch.tensor(mask,      dtype=torch.long),
        }


# ── Instantiate datasets ──────────────────────────────────────────────────────
print("Building RL datasets...")
train_rl = WOMDOfflineRLDataset(train_ds, max_scenarios=500)
val_rl   = WOMDOfflineRLDataset(val_ds,   max_scenarios=100)

train_loader = DataLoader(train_rl, batch_size=CFG["DT_BATCH"], shuffle=True,
                          num_workers=0, pin_memory=(DEVICE.type=="cuda"))
val_loader   = DataLoader(val_rl,   batch_size=CFG["DT_BATCH"], shuffle=False,
                          num_workers=0)

print(f"  Train samples: {len(train_rl):,}  |  Val samples: {len(val_rl):,}")


class DecisionTransformer(nn.Module):
    """
    Decision Transformer — Chen et al., 2021 (arXiv:2106.01345).

    Args:
        state_dim    : dimensionality of state vector
        act_dim      : dimensionality of action vector
        hidden_size  : GPT-2 d_model
        max_length   : context window (number of timesteps, not tokens)
        max_ep_len   : maximum episode length (for timestep embedding)
        n_layer      : GPT-2 number of transformer blocks
        n_head       : GPT-2 number of attention heads
        dropout      : residual & attention dropout
    """

    def __init__(self, state_dim: int, act_dim: int,
                 hidden_size: int = 128, max_length: int = 20,
                 max_ep_len: int = 200, n_layer: int = 3,
                 n_head: int = 1, dropout: float = 0.1):
        super().__init__()
        self.state_dim   = state_dim
        self.act_dim     = act_dim
        self.hidden_size = hidden_size
        self.max_length  = max_length

        config = GPT2Config(
            vocab_size=1,            # unused (embedding-only mode)
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * hidden_size,
            activation_function="relu",
            n_positions=3 * max_length + 2,   # 3 tokens per timestep
            resid_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2Model(config)

        # Input projections
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return   = nn.Linear(1,         hidden_size)
        self.embed_state    = nn.Linear(state_dim, hidden_size)
        self.embed_action   = nn.Linear(act_dim,   hidden_size)
        self.embed_ln       = nn.LayerNorm(hidden_size)

        # Output head
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh(),
        )

    def forward(self, states, actions, returns_to_go,
                timesteps, attention_mask=None):
        """
        states        : (B, T, state_dim)
        actions       : (B, T, act_dim)
        returns_to_go : (B, T, 1)
        timesteps     : (B, T)   long
        attention_mask: (B, T)   long, optional

        Returns
        -------
        action_preds  : (B, T, act_dim)
        """
        B, T = states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.long,
                                        device=states.device)

        # Embed each modality
        t_emb   = self.embed_timestep(timesteps)                  # (B, T, H)
        s_emb   = self.embed_state(states)    + t_emb             # (B, T, H)
        a_emb   = self.embed_action(actions)  + t_emb             # (B, T, H)
        r_emb   = self.embed_return(returns_to_go) + t_emb        # (B, T, H)

        # Interleave: (R₁, s₁, a₁, R₂, s₂, a₂, …)
        stacked = torch.stack([r_emb, s_emb, a_emb], dim=2)       # (B, T, 3, H)
        stacked = self.embed_ln(stacked.reshape(B, 3 * T, self.hidden_size))

        # Build 3T attention mask
        attn_3T = torch.stack([attention_mask]*3, dim=2).reshape(B, 3 * T)

        out = self.transformer(
            inputs_embeds=stacked,
            attention_mask=attn_3T,
        ).last_hidden_state                                        # (B, 3T, H)

        # Extract predictions at state token positions (index 1, 4, 7, …)
        out = out.reshape(B, T, 3, self.hidden_size)               # (B, T, 3, H)
        action_preds = self.predict_action(out[:, :, 1, :])        # (B, T, act_dim)
        return action_preds

    @torch.no_grad()
    def get_action(self, states, actions, returns_to_go, timesteps):
        """
        Autoregressive inference: return action for the last timestep.
        All inputs are 1D (no batch) or have batch dim = 1.
        """
        states        = states.reshape(1, -1, self.state_dim)
        actions       = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps     = timesteps.reshape(1, -1)

        # Trim context to max_length
        if self.max_length:
            states        = states[:, -self.max_length:]
            actions       = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps     = timesteps[:, -self.max_length:]

        action_preds = self.forward(states, actions, returns_to_go,
                                    timesteps, attention_mask=None)
        return action_preds[0, -1]   # (act_dim,)


# ── Instantiate ───────────────────────────────────────────────────────────────
dt_model = DecisionTransformer(
    state_dim   = CFG["STATE_DIM"],
    act_dim     = CFG["ACT_DIM"],
    hidden_size = CFG["DT_HIDDEN"],
    max_length  = CFG["CONTEXT_LEN"],
    max_ep_len  = CFG["PRED_HORIZON"] + 10,
    n_layer     = CFG["DT_N_LAYER"],
    n_head      = CFG["DT_N_HEAD"],
    dropout     = CFG["DT_DROPOUT"],
).to(DEVICE)

n_params = sum(p.numel() for p in dt_model.parameters() if p.requires_grad)
print(f"Decision Transformer  |  params: {n_params:,}")
print(dt_model)


import math as _math
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        s   = batch["states"].to(device)           # (B, CL, sd)
        a   = batch["actions"].to(device)           # (B, CL, ad)
        rtg = batch["returns_to_go"].to(device)     # (B, CL, 1)
        ts  = batch["timesteps"].to(device)         # (B, CL)
        msk = batch["attention_mask"].to(device)    # (B, CL)

        # Teacher forcing: predict a_t from (R_t, s_t)
        a_pred = model(s, a, rtg, ts, msk)         # (B, CL, ad)

        # MSE loss, only on valid (non-padded) positions — divide by valid count
        valid_mask = msk.unsqueeze(-1).float()      # (B, CL, 1)
        loss = (F.mse_loss(a_pred, a, reduction='none') * valid_mask).sum() \
               / (valid_mask.sum() * model.act_dim + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        s   = batch["states"].to(device)
        a   = batch["actions"].to(device)
        rtg = batch["returns_to_go"].to(device)
        ts  = batch["timesteps"].to(device)
        msk = batch["attention_mask"].to(device)
        a_pred = model(s, a, rtg, ts, msk)
        valid  = msk.unsqueeze(-1).float()
        loss   = (F.mse_loss(a_pred, a, reduction='none') * valid).sum() \
                 / (valid.sum() * model.act_dim + 1e-8)
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


# ── Training ──────────────────────────────────────────────────────────────────
dt_optimizer = torch.optim.Adam(dt_model.parameters(), lr=CFG["DT_LR"],
                                weight_decay=1e-4)
dt_scheduler = CosineAnnealingLR(dt_optimizer, T_max=CFG["DT_EPOCHS"], eta_min=1e-6)

dt_train_losses, dt_val_losses = [], []

print(f"Training Decision Transformer for {CFG['DT_EPOCHS']} epochs "
      f"(batch={CFG['DT_BATCH']}, device={DEVICE})")
print("-" * 60)

for epoch in range(1, CFG["DT_EPOCHS"] + 1):
    t0 = time.time()
    tr_loss = train_one_epoch(dt_model, train_loader, dt_optimizer, DEVICE)
    va_loss = evaluate(dt_model, val_loader, DEVICE)
    dt_scheduler.step()
    dt_train_losses.append(tr_loss)
    dt_val_losses.append(va_loss)
    elapsed = time.time() - t0
    if epoch == 1 or epoch % 5 == 0 or epoch == CFG["DT_EPOCHS"]:
        print(f"  Epoch {epoch:3d}/{CFG['DT_EPOCHS']} | "
              f"train={tr_loss:.4f}  val={va_loss:.4f} | {elapsed:.1f}s")

# ── Loss curves ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dt_train_losses, label="Train MSE", color="#1f77b4")
ax.plot(dt_val_losses,   label="Val MSE",   color="#ff7f0e")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("Decision Transformer — Training Curves")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dt_training_curves.png", dpi=120)
plt.show()

# ── Save checkpoint ───────────────────────────────────────────────────────────
torch.save({"model": dt_model.state_dict(),
            "cfg":   CFG,
            "epoch": CFG["DT_EPOCHS"]},
           "dt_checkpoint.pt")
print("Checkpoint saved → dt_checkpoint.pt")


@torch.no_grad()
def rollout_dt(model, initial_state: np.ndarray,
               desired_rtg: float = -0.8,
               horizon: int = None) -> np.ndarray:
    """
    Autoregressively generate a trajectory using the Decision Transformer.

    Returns predicted actions as a numpy array of shape [horizon, act_dim].
    """
    horizon = horizon or CFG["PRED_HORIZON"]
    model.eval()

    states        = [torch.zeros(CFG["STATE_DIM"])]
    states[0]     = torch.tensor(initial_state, dtype=torch.float32)
    actions       = [torch.zeros(CFG["ACT_DIM"])]
    returns_to_go = [torch.tensor([desired_rtg], dtype=torch.float32)]
    timesteps     = [torch.tensor([0], dtype=torch.long)]

    predicted = []

    for t in range(1, horizon):
        s   = torch.stack(states).unsqueeze(0).to(DEVICE)
        a   = torch.stack(actions).unsqueeze(0).to(DEVICE)
        rtg = torch.stack(returns_to_go).unsqueeze(0).to(DEVICE)
        ts  = torch.cat(timesteps).unsqueeze(0).to(DEVICE)

        act = model.get_action(s, a, rtg, ts).cpu()
        predicted.append(act.numpy())

        # Step: update state with predicted action (simple integration)
        new_s = states[-1].clone()
        new_s[0] += act[0]  # Δx / POS_SCALE
        new_s[1] += act[1]  # Δy / POS_SCALE

        states.append(new_s)
        actions.append(act)
        # Consume the step reward: RTG increases toward 0 as we travel
        step_disp = math.sqrt((act[0].item() * ACT_SCALE)**2 +
                               (act[1].item() * ACT_SCALE)**2)
        desired_rtg = min(0.0, desired_rtg + step_disp / CFG["RTG_SCALE"])
        returns_to_go.append(torch.tensor([desired_rtg], dtype=torch.float32))
        timesteps.append(torch.tensor([t]))

    return np.array(predicted)   # [horizon-1, act_dim]


# ── Demo rollout ──────────────────────────────────────────────────────────────
sample_state  = train_rl[0]["states"][0].numpy()   # take first state from dataset
dt_pred_traj  = rollout_dt(dt_model, sample_state, desired_rtg=-0.8)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(dt_pred_traj[:, 0] * ACT_SCALE, label="Δx", color="#1f77b4")
axes[0].plot(dt_pred_traj[:, 1] * ACT_SCALE, label="Δy", color="#ff7f0e")
axes[0].set_title("DT — Predicted Actions (Δx, Δy)"); axes[0].legend(); axes[0].grid(alpha=0.3)

# Reconstruct trajectory from actions
traj_x = np.cumsum(dt_pred_traj[:, 0]) * ACT_SCALE
traj_y = np.cumsum(dt_pred_traj[:, 1]) * ACT_SCALE
axes[1].plot(traj_x, traj_y, "o-", color="#2ca02c", markersize=3)
axes[1].set_title("DT — Reconstructed Trajectory"); axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("y (m)")
axes[1].grid(alpha=0.3); axes[1].set_aspect("equal")

plt.tight_layout()
plt.savefig("dt_inference.png", dpi=120)
plt.show()


class TrajectoryTransformer(nn.Module):
    """
    Simplified Trajectory Transformer — Janner et al., 2021 (arXiv:2106.02039).

    Discretises states and actions, then trains a causal LM on the token sequence.
    """

    def __init__(self, state_dim: int, act_dim: int,
                 hidden_size: int = 128, n_bins: int = 64,
                 n_layer: int = 3, n_head: int = 1,
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.state_dim  = state_dim
        self.act_dim    = act_dim
        self.n_bins     = n_bins

        # Token ranges:
        #   state dim d:  d * n_bins  …  (d+1) * n_bins - 1
        #   action dim d: (state_dim + d) * n_bins  …  (state_dim + d+1) * n_bins - 1
        #   reward:       (state_dim + act_dim) * n_bins
        self.state_offset  = 0
        self.action_offset = state_dim * n_bins
        self.reward_token  = (state_dim + act_dim) * n_bins
        self.vocab_size    = self.reward_token + 1

        config = GPT2Config(
            vocab_size=self.vocab_size,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * hidden_size,
            n_positions=max_seq_len,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            activation_function="gelu",
        )
        self.transformer = GPT2Model(config)
        self.lm_head     = nn.Linear(hidden_size, self.vocab_size, bias=False)
        # Tie weights (as in standard LM)
        self.transformer.wte.weight = self.lm_head.weight

        # Discretisation bin edges (fitted from training data)
        self._state_bins  = None   # np array [state_dim, n_bins+1]
        self._action_bins = None   # np array [act_dim,   n_bins+1]

    # ── Discretisation ────────────────────────────────────────────────────────

    def fit_discretisation(self, states_all: np.ndarray,
                           actions_all: np.ndarray):
        """
        Compute per-dimension percentile bin edges.
        states_all : (N, state_dim)
        actions_all: (N, act_dim)
        """
        def _edges(data, dim):
            edges = []
            for d in range(dim):
                col  = data[:, d]
                q    = np.quantile(col, np.linspace(0, 1, self.n_bins + 1))
                q    = np.unique(q)   # remove duplicates
                edges.append(q)
            return edges

        self._state_bins  = _edges(states_all,  self.state_dim)
        self._action_bins = _edges(actions_all, self.act_dim)
        print(f"  Discretisation fitted on {len(states_all):,} samples per dim")

    def _discretise_dim(self, val: float, edges: np.ndarray) -> int:
        """Map a scalar to a bin index ∈ [0, n_bins-1]."""
        idx = int(np.searchsorted(edges, val, side="right")) - 1
        return int(np.clip(idx, 0, len(edges) - 2))

    def tokenise(self, states: np.ndarray,
                 actions: np.ndarray) -> np.ndarray:
        """
        Convert one trajectory to a token sequence.
        states:  (T, state_dim)
        actions: (T, act_dim)
        Returns token_ids: (T * (state_dim + act_dim + 1),)
        """
        assert self._state_bins is not None, "Call fit_discretisation first"
        T = states.shape[0]
        tokens = []
        for t in range(T):
            for d in range(self.state_dim):
                edge = self._state_bins[d]
                idx  = self._discretise_dim(states[t, d], edge)
                tokens.append(self.state_offset + d * self.n_bins + idx)
            for d in range(self.act_dim):
                edge = self._action_bins[d]
                idx  = self._discretise_dim(actions[t, d], edge)
                tokens.append(self.action_offset + d * self.n_bins + idx)
            tokens.append(self.reward_token)   # reward placeholder
        return np.array(tokens, dtype=np.int64)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, token_ids, attention_mask=None):
        """
        token_ids:      (B, L)  long
        attention_mask: (B, L)  long (optional)
        Returns logits: (B, L, vocab_size)
        """
        emb  = self.transformer(input_ids=token_ids,
                                 attention_mask=attention_mask)
        logits = self.lm_head(emb.last_hidden_state)
        return logits   # (B, L, V)

    # ── Beam search inference ─────────────────────────────────────────────────

    @torch.no_grad()
    def beam_search_actions(self, state_tokens: torch.Tensor,
                             beam_size: int = 4) -> torch.Tensor:
        """
        Given the discretised state tokens for the current timestep,
        use beam search to generate the most likely action tokens.

        state_tokens: (L,) long tensor  — full context + current state tokens
        Returns      : (act_dim,)  long tensor  (best beam action token per dim)
        """
        self.eval()
        # Truncate context so sequence never exceeds n_positions during beam search
        max_ctx = self.transformer.config.n_positions - self.act_dim - 2
        if state_tokens.shape[0] > max_ctx:
            state_tokens = state_tokens[-max_ctx:]
        context = state_tokens.unsqueeze(0)   # (1, L)

        # Initialise beams: list of (score, token_sequence)
        beams = [(0.0, context)]

        for step in range(self.act_dim):
            new_beams = []
            for score, seq in beams:
                seq = seq.to(DEVICE)
                logits = self.forward(seq)[:, -1, :]  # (1, V)

                # Only consider action tokens for this dimension
                act_start = self.action_offset + step * self.n_bins
                act_end   = act_start + self.n_bins
                act_logits = logits[0, act_start:act_end]       # (n_bins,)
                log_probs  = F.log_softmax(act_logits, dim=-1)

                topk_vals, topk_idx = log_probs.topk(beam_size)
                for log_p, tok_local in zip(topk_vals, topk_idx):
                    tok_global = torch.tensor(
                        [[act_start + tok_local.item()]], dtype=torch.long)
                    new_seq = torch.cat([seq.cpu(), tok_global], dim=1)
                    new_beams.append((score + log_p.item(), new_seq))

            # Keep top beam_size beams
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]

        # Extract action tokens from best beam (the last act_dim tokens appended)
        best_seq = beams[0][1][0]              # (L + act_dim,)
        act_tokens = best_seq[-self.act_dim:]  # (act_dim,)
        return act_tokens


# ── Instantiate ───────────────────────────────────────────────────────────────
tt_model = TrajectoryTransformer(
    state_dim   = CFG["STATE_DIM"],
    act_dim     = CFG["ACT_DIM"],
    hidden_size = CFG["TT_HIDDEN"],
    n_bins      = CFG["TT_N_BINS"],
    n_layer     = CFG["TT_N_LAYER"],
    n_head      = CFG["TT_N_HEAD"],
    dropout     = CFG["TT_DROPOUT"],
    # n_positions must fit the full rollout context:
    # (CONTEXT_LEN + PRED_HORIZON) * tokens_per_step  with headroom
    max_seq_len = (CFG["CONTEXT_LEN"] + CFG["PRED_HORIZON"]) *                   (CFG["STATE_DIM"] + CFG["ACT_DIM"] + 1) + 16,
).to(DEVICE)

n_params = sum(p.numel() for p in tt_model.parameters() if p.requires_grad)
print(f"Trajectory Transformer  |  params: {n_params:,}")


# ── Build TT-specific token dataset ──────────────────────────────────────────
print("Fitting TT discretisation on training data...")

# Collect all states & actions for fitting bin edges
all_states  = np.concatenate([train_rl[i]["states"].numpy()
                               for i in range(min(1000, len(train_rl)))], axis=0)
all_actions = np.concatenate([train_rl[i]["actions"].numpy()
                               for i in range(min(1000, len(train_rl)))], axis=0)
tt_model.fit_discretisation(all_states, all_actions)


class TTTokenDataset(Dataset):
    """Converts RL dataset samples to TT token sequences."""
    def __init__(self, rl_dataset, max_len: int = 512):
        self.rl_dataset = rl_dataset
        self.max_len    = max_len

    def __len__(self):
        return len(self.rl_dataset)

    def __getitem__(self, idx):
        sample  = self.rl_dataset[idx]
        states  = sample["states"].numpy()   # (CL, sd)
        actions = sample["actions"].numpy()  # (CL, ad)
        mask    = sample["attention_mask"].numpy()  # (CL,)

        T = int(mask.sum())
        tokens = tt_model.tokenise(states[:T], actions[:T])
        tokens = tokens[:self.max_len]

        input_ids  = torch.tensor(tokens[:-1], dtype=torch.long)
        labels     = torch.tensor(tokens[1:],  dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}


def tt_collate(batch):
    """Pad sequences in a batch to equal length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for b in batch:
        L    = b["input_ids"].shape[0]
        pad  = max_len - L
        input_ids_list.append(F.pad(b["input_ids"], (0, pad), value=0))
        labels_list.append(F.pad(b["labels"],    (0, pad), value=-100))
        mask_list.append(torch.cat([torch.ones(L), torch.zeros(pad)]).long())
    return {
        "input_ids":      torch.stack(input_ids_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


tt_train_ds = TTTokenDataset(train_rl)
tt_val_ds   = TTTokenDataset(val_rl)
tt_train_loader = DataLoader(tt_train_ds, batch_size=CFG["TT_BATCH"],
                             shuffle=True, collate_fn=tt_collate, num_workers=0)
tt_val_loader   = DataLoader(tt_val_ds,   batch_size=CFG["TT_BATCH"],
                             shuffle=False, collate_fn=tt_collate, num_workers=0)


# ── Training loop ─────────────────────────────────────────────────────────────
tt_optimizer = torch.optim.Adam(tt_model.parameters(), lr=CFG["TT_LR"],
                                weight_decay=1e-4)
tt_scheduler = CosineAnnealingLR(tt_optimizer, T_max=CFG["TT_EPOCHS"], eta_min=1e-6)

tt_train_losses, tt_val_losses = [], []

print(f"Training Trajectory Transformer for {CFG['TT_EPOCHS']} epochs "
      f"(batch={CFG['TT_BATCH']}, device={DEVICE})")
print("-" * 60)


def tt_train_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        lbls = batch["labels"].to(device)
        msk  = batch["attention_mask"].to(device)
        logits = model(ids, attention_mask=msk)            # (B, L, V)
        loss   = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                                  lbls.reshape(-1),
                                  ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def tt_eval_epoch(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        lbls = batch["labels"].to(device)
        msk  = batch["attention_mask"].to(device)
        logits = model(ids, attention_mask=msk)
        loss   = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                                  lbls.reshape(-1), ignore_index=-100)
        total += loss.item(); n += 1
    return total / max(n, 1)


for epoch in range(1, CFG["TT_EPOCHS"] + 1):
    t0 = time.time()
    tr = tt_train_epoch(tt_model, tt_train_loader, tt_optimizer, DEVICE)
    va = tt_eval_epoch(tt_model, tt_val_loader, DEVICE)
    tt_scheduler.step()
    tt_train_losses.append(tr)
    tt_val_losses.append(va)
    elapsed = time.time() - t0
    if epoch == 1 or epoch % 5 == 0 or epoch == CFG["TT_EPOCHS"]:
        print(f"  Epoch {epoch:3d}/{CFG['TT_EPOCHS']} | "
              f"train={tr:.4f}  val={va:.4f} | {elapsed:.1f}s")

# ── Loss curves ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(tt_train_losses, label="Train CE Loss", color="#1f77b4")
ax.plot(tt_val_losses,   label="Val CE Loss",   color="#ff7f0e")
ax.set_title("Trajectory Transformer — Training Curves")
ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("tt_training_curves.png", dpi=120); plt.show()

torch.save({"model": tt_model.state_dict(),
            "state_bins":  tt_model._state_bins,
            "action_bins": tt_model._action_bins,
            "cfg": CFG},
           "tt_checkpoint.pt")
print("Checkpoint saved → tt_checkpoint.pt")


@torch.no_grad()
def rollout_tt(model, initial_states: np.ndarray,
               horizon: int = None, beam_size: int = 4) -> np.ndarray:
    """
    Autoregressively generate a trajectory using the Trajectory Transformer.
    Uses beam search over action tokens at each step.

    initial_states: (context_len, state_dim) numpy array
    Returns:        (horizon, act_dim) predicted action array
    """
    horizon = horizon or CFG["PRED_HORIZON"]
    model.eval()

    # Tokenise the initial state context
    dummy_acts = np.zeros((len(initial_states), CFG["ACT_DIM"]), dtype=np.float32)
    context_tokens = model.tokenise(initial_states, dummy_acts)
    # Drop dummy action tokens; keep only state tokens from the last step
    stride      = CFG["STATE_DIM"] + CFG["ACT_DIM"] + 1   # tokens per timestep
    last_state_tokens = context_tokens[-stride: -stride + CFG["STATE_DIM"]]

    predicted_actions = []
    current_context   = torch.tensor(
        context_tokens[:-stride + CFG["STATE_DIM"]],   # strip dummy actions/reward
        dtype=torch.long).unsqueeze(0).to(DEVICE)

    for step in range(horizon):
        state_tok = torch.tensor(last_state_tokens, dtype=torch.long).to(DEVICE)
        act_toks  = model.beam_search_actions(
            torch.cat([current_context[0], state_tok]),
            beam_size=beam_size
        )

        # De-tokenise actions back to continuous values
        act = np.zeros(CFG["ACT_DIM"], dtype=np.float32)
        for d in range(CFG["ACT_DIM"]):
            local_idx = act_toks[d].item() - (model.action_offset + d * model.n_bins)
            local_idx = int(np.clip(local_idx, 0, model.n_bins - 1))
            edges = model._action_bins[d]
            # Use bin centre
            if local_idx + 1 < len(edges):
                act[d] = (edges[local_idx] + edges[local_idx + 1]) / 2
            else:
                act[d] = edges[-1]
        predicted_actions.append(act)

        # Build next state (simple integration) and append to context
        new_state   = initial_states[-1].copy()
        new_state[0] = np.clip(new_state[0] + act[0], -1.5, 1.5)
        new_state[1] = np.clip(new_state[1] + act[1], -1.5, 1.5)
        new_state_tokens = np.array(
            [model.state_offset + d * model.n_bins +
             model._discretise_dim(new_state[d], model._state_bins[d])
             for d in range(model.state_dim)], dtype=np.int64)
        last_state_tokens = new_state_tokens
        current_context = torch.cat([
            current_context,
            act_toks.unsqueeze(0).to(DEVICE),
            torch.tensor([[model.reward_token]], dtype=torch.long).to(DEVICE),
            torch.tensor([new_state_tokens], dtype=torch.long).to(DEVICE),
        ], dim=1)

    return np.array(predicted_actions)    # (horizon, act_dim)


# ── Demo rollout ──────────────────────────────────────────────────────────────
init_states = train_rl[0]["states"].numpy()[:CFG["CONTEXT_LEN"]]
tt_pred_traj = rollout_tt(tt_model, init_states, horizon=CFG["PRED_HORIZON"],
                          beam_size=CFG["TT_BEAM_SIZE"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(tt_pred_traj[:, 0] * ACT_SCALE, label="Δx", color="#1f77b4")
axes[0].plot(tt_pred_traj[:, 1] * ACT_SCALE, label="Δy", color="#ff7f0e")
axes[0].set_title("TT — Predicted Actions (Δx, Δy)"); axes[0].legend(); axes[0].grid(alpha=0.3)

traj_x = np.cumsum(tt_pred_traj[:, 0]) * ACT_SCALE
traj_y = np.cumsum(tt_pred_traj[:, 1]) * ACT_SCALE
axes[1].plot(traj_x, traj_y, "s-", color="#9467bd", markersize=3)
axes[1].set_title("TT — Reconstructed Trajectory (Beam Search)")
axes[1].set_xlabel("x (m)"); axes[1].set_ylabel("y (m)")
axes[1].grid(alpha=0.3); axes[1].set_aspect("equal")

plt.tight_layout(); plt.savefig("tt_inference.png", dpi=120); plt.show()


def compute_ade_fde(pred_actions: np.ndarray,
                    gt_actions:   np.ndarray) -> tuple:
    """
    pred_actions, gt_actions: (T, 2)  (Δx, Δy) in normalised units
    Returns (ADE, FDE) in metres.
    """
    pred_pos = np.cumsum(pred_actions, axis=0) * ACT_SCALE
    gt_pos   = np.cumsum(gt_actions,   axis=0) * ACT_SCALE
    dists    = np.linalg.norm(pred_pos - gt_pos, axis=1)   # (T,)
    ade      = dists.mean()
    fde      = dists[-1]
    return float(ade), float(fde)


def constant_velocity_baseline(initial_state: np.ndarray,
                                horizon: int) -> np.ndarray:
    """Predict constant-velocity actions in AV-local frame (matches training)."""
    evx = initial_state[4] * VEL_SCALE * CFG["DT_SEC"]   # world-frame vx * dt
    evy = initial_state[5] * VEL_SCALE * CFG["DT_SEC"]   # world-frame vy * dt
    cos_h, sin_h = initial_state[2], initial_state[3]
    xl = ( evx * cos_h + evy * sin_h) / ACT_SCALE
    yl = (-evx * sin_h + evy * cos_h) / ACT_SCALE
    return np.tile([xl, yl], (horizon, 1)).astype(np.float32)


# ── Evaluate on validation set (up to 100 samples) ───────────────────────────
N_EVAL = min(100, len(val_rl))
results = []

print(f"Evaluating on {N_EVAL} validation trajectories...")

for i in range(N_EVAL):
    sample    = val_rl[i]
    gt_act    = sample["actions"].numpy()                          # (CL, 2)
    init_s    = sample["states"].numpy()[0]                        # (state_dim,)
    ts        = sample["timesteps"].numpy()                        # (CL,)

    horizon   = CFG["PRED_HORIZON"]

    # Decision Transformer
    dt_pred   = rollout_dt(dt_model, init_s, desired_rtg=-0.8, horizon=horizon)
    dt_ade, dt_fde = compute_ade_fde(dt_pred, gt_act[:len(dt_pred)])

    # Trajectory Transformer
    init_ctx  = sample["states"].numpy()[:CFG["CONTEXT_LEN"]]
    tt_pred   = rollout_tt(tt_model, init_ctx, horizon=horizon, beam_size=2)
    tt_ade, tt_fde = compute_ade_fde(tt_pred, gt_act[:len(tt_pred)])

    # Constant-velocity baseline
    cv_pred   = constant_velocity_baseline(init_s, horizon)
    cv_ade, cv_fde = compute_ade_fde(cv_pred, gt_act[:len(cv_pred)])

    results.append({
        "idx": i,
        "DT_ADE": dt_ade, "DT_FDE": dt_fde,
        "TT_ADE": tt_ade, "TT_FDE": tt_fde,
        "CV_ADE": cv_ade, "CV_FDE": cv_fde,
    })

results_df = pd.DataFrame(results)
print("\n── Evaluation Summary ──────────────────────────────────────────────────")
print(results_df[["DT_ADE","DT_FDE","TT_ADE","TT_FDE","CV_ADE","CV_FDE"]].describe().round(3))

# ── Comparison bar chart ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
models     = ["Decision\nTransformer", "Trajectory\nTransformer", "Const.\nVelocity"]
ade_means  = [results_df["DT_ADE"].mean(), results_df["TT_ADE"].mean(), results_df["CV_ADE"].mean()]
fde_means  = [results_df["DT_FDE"].mean(), results_df["TT_FDE"].mean(), results_df["CV_FDE"].mean()]
colors     = ["#1f77b4", "#9467bd", "#7f7f7f"]

bars_ade = axes[0].bar(models, ade_means, color=colors, edgecolor="white", width=0.5)
axes[0].set_title("minADE@1 (lower is better)"); axes[0].set_ylabel("ADE (m)")
for b in bars_ade:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                 f"{b.get_height():.2f}", ha="center", fontsize=9)

bars_fde = axes[1].bar(models, fde_means, color=colors, edgecolor="white", width=0.5)
axes[1].set_title("minFDE@1 (lower is better)"); axes[1].set_ylabel("FDE (m)")
for b in bars_fde:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                 f"{b.get_height():.2f}", ha="center", fontsize=9)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Model Comparison on WOMD Validation Set", fontsize=12)
plt.tight_layout(); plt.savefig("eval_comparison.png", dpi=120); plt.show()

# ── Trajectory visualisation side by side ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Sample Trajectory Comparison (val idx=0)", fontsize=12)

sample0   = val_rl[0]
gt_traj   = np.cumsum(sample0["actions"].numpy(), axis=0) * ACT_SCALE
dt_traj   = np.cumsum(dt_pred, axis=0) * ACT_SCALE
tt_traj   = np.cumsum(tt_pred, axis=0) * ACT_SCALE
cv_traj   = np.cumsum(constant_velocity_baseline(val_rl[0]["states"].numpy()[0],
                                                   CFG["PRED_HORIZON"]), axis=0) * ACT_SCALE

for ax, pred_traj, title, color in zip(
    axes,
    [dt_traj, tt_traj, cv_traj],
    ["Decision Transformer", "Trajectory Transformer", "Const. Velocity"],
    ["#1f77b4", "#9467bd", "#7f7f7f"]
):
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], "k--", lw=2, label="Ground Truth")
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], "o-", color=color,
            markersize=3, lw=1.5, label=title)
    ax.set_title(title); ax.legend(fontsize=7)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3); ax.set_aspect("equal")

plt.tight_layout(); plt.savefig("trajectory_comparison.png", dpi=120); plt.show()

