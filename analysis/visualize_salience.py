"""Animate the camera view beside its salience map, step by step.

The salience map is not recorded in the detailed stats, but the VOCUS2
strategy is deterministic given the sensor's rgba and depth, which are
recorded — so this script recomputes each frame's map with the same
configuration the potato experiments use (see
``conf/monty/sensor_module/*_camera_dist_vocus2_slic.yaml``) and shows it
next to the frame-plus-segmentation view from ``visualize_3d``.

Usage::

    uv run python analysis/visualize_salience.py [exp_dir] [--episode N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from detailed_stats import available_episodes, load_episode_stats
from visualize_3d import DEFAULT_EXP_DIR, SMTelemetry
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.salience.strategies.vocus2.vocus2 import (
    Vocus2,
    Vocus2SalienceConfig,
)

# Mirrors the sensor-module yaml used by the potato experiments.
POTATO_VOCUS2_CONFIG = Vocus2SalienceConfig(
    center_sigma=1.0,
    surround_sigma=6.5,
    n_scales=2,
    max_octaves=5,
    use_depth=True,
    use_orientation=False,
)


def compute_salience_maps(stats: dict, sensor_module_id: str) -> list[np.ndarray]:
    """Recompute each frame's VOCUS2 salience map from recorded observations.

    Args:
        stats: Loaded episode stats.
        sensor_module_id: Which sensor module's observations to use.

    Returns:
        One ``(H, W)`` salience map per recorded frame.
    """
    strategy = Vocus2.from_config(POTATO_VOCUS2_CONFIG)
    ctx = RuntimeContext(rng=np.random.RandomState(0))
    maps = []
    for obs in stats[sensor_module_id]["raw_observations"]:
        rgba = np.array(obs["rgba"], dtype=np.uint8)
        depth = np.array(obs["depth"], dtype=np.float64)
        maps.append(np.asarray(strategy(ctx=ctx, rgba=rgba, depth=depth)))
    return maps


def create_salience_animation(
    exp_dir: Path,
    episode: int = 0,
    sensor_module_id: str | int = 1,
    fps: int = 2,
    interval: int = 500,
) -> Path:
    """Animate the sensor view beside its recomputed salience map.

    Args:
        exp_dir: Experiment directory.
        episode: Episode to visualize.
        sensor_module_id: Sensor module to read; SM_1 is the SalienceSM.
        fps: Frames per second of the saved gif.
        interval: Milliseconds between animation frames.

    Returns:
        Path to the saved gif.
    """
    stats = load_episode_stats(exp_dir, episode=episode)
    sm = SMTelemetry(stats, sensor_module_id)

    recorded = stats[sm.sensor_module_id].get("salience_maps", [])
    if recorded and all(m is not None for m in recorded):
        print(f"Using {len(recorded)} recorded salience maps.")
        salience_maps = [np.asarray(m, dtype=float) for m in recorded]
    else:
        # Stats recorded before salience maps rode in telemetry: recompute
        # with the potato configuration (faithful only if it matches the run).
        print(f"No recorded salience maps; recomputing for {sm.n_frames} frames ...")
        salience_maps = compute_salience_maps(stats, sm.sensor_module_id)

    # One fixed colour scale across the episode so brightness is comparable
    # between frames.
    vmin = min(float(m.min()) for m in salience_maps)
    vmax = max(float(m.max()) for m in salience_maps)
    if vmax == vmin:
        vmax = vmin + 1.0

    n_frames = sm.n_frames
    fig = plt.figure(figsize=(13, 6))
    grid = fig.add_gridspec(1, 2, wspace=0.15)
    ax_image = fig.add_subplot(grid[0, 0])
    ax_salience = fig.add_subplot(grid[0, 1])
    fig.suptitle("Salience", fontsize=14, fontweight="bold")

    ax_image.axis("off")
    ax_salience.axis("off")

    image = ax_image.imshow(sm.overlay(0))
    heat = ax_salience.imshow(
        salience_maps[0], cmap="inferno", vmin=vmin, vmax=vmax
    )
    bar = plt.colorbar(heat, ax=ax_salience, fraction=0.046, pad=0.04)
    bar.set_label("Salience", rotation=270, labelpad=15)

    # Mark the fixation: the sensor patch is centred on what it fixates.
    height, width = sm.rgbas[0].shape[:2]
    for ax in (ax_image, ax_salience):
        ax.add_patch(
            Rectangle(
                (width // 2 - 1.5, height // 2 - 1.5),
                3,
                3,
                linewidth=1,
                edgecolor="white" if ax is ax_salience else "black",
                facecolor="none",
            )
        )

    def update_frame(step: int):
        image.set_data(sm.overlay(step))
        label = "Frame" if not sm.has_segmentation else "Frame + Segmentation"
        ax_image.set_title(f"{label} (Step {step}/{n_frames - 1})")
        heat.set_data(salience_maps[step])
        ax_salience.set_title(f"VOCUS2 salience (Step {step}/{n_frames - 1})")
        return [image, heat]

    anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=interval,
                         blit=True)

    visualizations_dir = exp_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    gif_path = visualizations_dir / f"salience_{episode}.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps))

    print(f"Animation saved to: {gif_path}")
    plt.close()

    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate the sensor view beside its VOCUS2 salience map."
    )
    parser.add_argument(
        "exp_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_EXP_DIR,
        help=f"Experiment output directory (default: {DEFAULT_EXP_DIR})",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode to animate (default: every recorded episode)",
    )
    parser.add_argument("--fps", type=int, default=2, help="GIF frames per second")
    args = parser.parse_args()

    episodes = (
        [args.episode]
        if args.episode is not None
        else available_episodes(args.exp_dir)
    )
    if not episodes:
        raise SystemExit(f"No detailed stats found under {args.exp_dir}")
    for episode in episodes:
        create_salience_animation(args.exp_dir, episode=episode, fps=args.fps)


if __name__ == "__main__":
    main()
