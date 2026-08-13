"""Animate the SalienceSM's per-step 3D salience change.

Every step the SalienceSM voxelizes the on-object salience it just computed --
one mean salience per voxel -- and diffs that map against the previous step's,
keeping the voxels that moved by more than its diff threshold (see
``SalienceSM._step_salience3d``). Those surviving voxels are the module's change
signal: where in 3D the scene just changed.

Built with ``save_salience3d_diff: true``, the module records them, and they
ride in the detailed stats under the sensor module::

    stats["SM_9"] = {
        "raw_observations": [{"rgba": (H, W, 4), ...}, ...],
        "segmentation_maps": [(H, W) mask or None, ...],
        ...,
        "salience3d_voxel_size": float,
        "salience3d_diff_threshold": float,
        "salience3d_diffs": [                     # one entry per recorded step
            {"voxels": (V, 3) int, "diff": (V,) float},
            ...,
        ],
    }

The module records a step only while it is not exploring, so these entries line
up index for index with ``raw_observations`` and ``segmentation_maps``. The
first entry of an episode is empty by construction -- there is no previous map
to diff against -- as is the first after a reset.

The figure pairs the sensor view (segmentation tinted green) with a 3D scatter
of the changed voxels coloured by how far they moved, and a trace of the change
over the episode with a cursor on the current step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from detailed_stats import available_episodes, load_episode_stats
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from visualize_3d import (
    DEFAULT_EXP_DIR,
    OKABE_ITO,
    VOXEL_CENTRE_OFFSET,
    SMTelemetry,
    _bounds,
)
from visualize_goal_filtering import interior_points

# Sequential map: the diff is a magnitude, so it reads from dark (barely over
# the threshold) to bright (the biggest change in the episode).
DIFF_CMAP = "magma"

COUNT_COLOR = OKABE_ITO[3]  # blue
TOTAL_COLOR = OKABE_ITO[0]  # orange


def find_salience_sm(stats: dict) -> str:
    """Find the sensor module that recorded the salience3d diff.

    Args:
        stats: Loaded episode stats.

    Returns:
        The module's key, e.g. ``SM_9``.

    Raises:
        SystemExit: If no sensor module recorded the diff, or the one that did
            saved no frames to draw it against.
    """
    recorded = [
        key
        for key, value in stats.items()
        if key.startswith("SM_")
        and isinstance(value, dict)
        and value.get("salience3d_diffs")
    ]
    if not recorded:
        raise SystemExit(
            "No sensor module in these stats recorded a salience3d diff. Re-run "
            "with save_salience3d_diff: true on the SalienceSM's telemetry."
        )

    sensor_module_id = recorded[0]
    if not stats[sensor_module_id].get("raw_observations"):
        raise SystemExit(
            f"{sensor_module_id} recorded a salience3d diff but no frames to show "
            "it against. Re-run with save_raw_obs: true on the SalienceSM."
        )
    return sensor_module_id


class Salience3dDiffTelemetry:
    """Plot-ready views over the SalienceSM's recorded salience3d diff.

    Attributes:
        sensor_module_id: The module the diff was read from.
        voxel_size: Edge length of a salience3d voxel in meters.
        threshold: The change a voxel had to exceed to be recorded.
        centres: Per step, the ``(N, 3)`` centres of the changed voxels.
        magnitudes: Per step, the ``(N,)`` absolute change of those voxels.
    """

    def __init__(self, stats: dict, sensor_module_id: str) -> None:
        """Read the salience3d diff out of loaded episode stats.

        Args:
            stats: Loaded episode stats.
            sensor_module_id: Which sensor module to read.
        """
        self.sensor_module_id = sensor_module_id
        sm = stats[sensor_module_id]

        self.voxel_size = sm["salience3d_voxel_size"]
        self.threshold = sm["salience3d_diff_threshold"]

        self.centres: list[np.ndarray] = []
        self.magnitudes: list[np.ndarray] = []
        for entry in sm.get("salience3d_diffs", []):
            # A step where nothing changed serializes as [], so reshape rather
            # than assume (V, 3).
            voxels = np.asarray(entry["voxels"], dtype=int).reshape(-1, 3)
            self.centres.append((voxels + VOXEL_CENTRE_OFFSET) * self.voxel_size)
            self.magnitudes.append(np.asarray(entry["diff"], dtype=float).ravel())

    @property
    def n_steps(self) -> int:
        """How many steps were recorded."""
        return len(self.centres)

    @property
    def has_diff(self) -> bool:
        """Whether any voxel anywhere in the episode cleared the threshold."""
        return any(len(magnitude) for magnitude in self.magnitudes)

    @property
    def counts(self) -> np.ndarray:
        """The number of changed voxels per step."""
        return np.array([len(magnitude) for magnitude in self.magnitudes])

    @property
    def totals(self) -> np.ndarray:
        """The summed absolute change per step."""
        return np.array([float(magnitude.sum()) for magnitude in self.magnitudes])

    def at(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the changed voxels of one step.

        Args:
            step: Which step to read.

        Returns:
            A ``(centres, magnitudes)`` pair, empty past the record.
        """
        if step >= self.n_steps:
            return np.empty((0, 3)), np.empty(0)
        return self.centres[step], self.magnitudes[step]

    def bounds_points(self) -> list[np.ndarray]:
        """Return every changed voxel centre, for axis limits.

        Returns:
            Point arrays spanning the episode's changes.
        """
        return [centres for centres in self.centres if len(centres)]

    def value_range(self) -> tuple[float, float]:
        """Return the colour scale spanning the episode's changes.

        Returns:
            A ``(vmin, vmax)`` pair; the threshold is the floor, since nothing
            below it was recorded.
        """
        populated = [m for m in self.magnitudes if len(m)]
        if not populated:
            return self.threshold, self.threshold * 2
        high = float(max(m.max() for m in populated))
        return self.threshold, max(high, self.threshold * 2)


def within_limits(
    points: np.ndarray,
    xlim: list[float],
    ylim: list[float],
    zlim: list[float],
) -> np.ndarray:
    """Return a mask of the points inside the axis limits.

    Matplotlib's 3D axes do not clip scatter points to the axes box, so without
    this, changes on a room's surfaces bleed past object-scale limits. A mask
    rather than a filtered copy, so the magnitudes can be cut alongside.

    Args:
        points: A ``(N, 3)`` array.
        xlim: X-axis limits.
        ylim: Y-axis limits.
        zlim: Z-axis limits.

    Returns:
        A ``(N,)`` boolean mask.
    """
    if not len(points):
        return np.zeros(0, dtype=bool)
    low = np.array([xlim[0], ylim[0], zlim[0]])
    high = np.array([xlim[1], ylim[1], zlim[1]])
    return ((points >= low) & (points <= high)).all(axis=1)


def create_salience3d_diff_animation(
    exp_dir: Path,
    episode: int = 0,
    sensor_module_id: str | int | None = None,
    fps: int = 2,
    interval: int = 500,
    marker_size: int = 12,
) -> Path:
    """Animate the sensor view beside the salience3d diff it produced.

    Args:
        exp_dir: Experiment directory.
        episode: Episode to visualize.
        sensor_module_id: Sensor module to read; found from the stats if None.
        fps: Frames per second of the saved gif.
        interval: Milliseconds between animation frames.
        marker_size: Scatter marker size in the 3D panel.

    Returns:
        Path to the saved gif.
    """
    stats = load_episode_stats(exp_dir, episode=episode)

    if sensor_module_id is None:
        sensor_module_id = find_salience_sm(stats)
    elif isinstance(sensor_module_id, int) or str(sensor_module_id).isdigit():
        sensor_module_id = f"SM_{sensor_module_id}"

    sm = SMTelemetry(stats, sensor_module_id)
    diffs = Salience3dDiffTelemetry(stats, sensor_module_id)

    if not diffs.has_diff:
        print(
            "No voxel changed by more than the module's threshold "
            f"({diffs.threshold}) in this episode."
        )

    n_frames = min(sm.n_frames, diffs.n_steps)
    xlim, ylim, zlim = _bounds(interior_points(diffs.bounds_points()))
    vmin, vmax = diffs.value_range()
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(17, 5.5))
    # The 3D panel carries the colour bar, so it gets the extra width.
    grid = fig.add_gridspec(1, 3, wspace=0.3, width_ratios=(1, 1.25, 1))
    ax_image = fig.add_subplot(grid[0, 0])
    ax_diff = fig.add_subplot(grid[0, 1], projection="3d")
    ax_trace = fig.add_subplot(grid[0, 2])

    fig.suptitle("SalienceSM 3D Salience Change", fontsize=14, fontweight="bold")
    # Along the bottom: the panel titles change every frame, and a caption up
    # by the suptitle collides with them.
    fig.text(
        0.5,
        0.02,
        f"{sensor_module_id} | voxel {diffs.voxel_size} m | "
        f"threshold {diffs.threshold}",
        ha="center",
        fontsize=8,
        color="dimgray",
    )

    def style_3d(ax) -> None:
        """Apply shared 3D styling. Re-applied after every ax.clear()."""
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_box_aspect([1, 1, 1])
        # Top-down: elev=90 looks along Z, azim=-90 orients X/Y conventionally.
        ax.view_init(elev=90, azim=-90)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ax_image.axis("off")
    style_3d(ax_diff)

    image = ax_image.imshow(sm.overlay(0))

    # Mark the fixation: the sensor patch is centred on what it fixates.
    height, width = sm.rgbas[0].shape[:2]
    ax_image.add_patch(
        Rectangle(
            (width // 2 - 1.5, height // 2 - 1.5),
            3,
            3,
            linewidth=1,
            edgecolor="black",
            facecolor="none",
        )
    )

    colour_bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=DIFF_CMAP),
        ax=ax_diff,
        fraction=0.03,
        pad=0.06,
        shrink=0.75,
    )
    colour_bar.set_label("|salience change|", fontsize=8)
    colour_bar.ax.tick_params(labelsize=7)

    steps = np.arange(diffs.n_steps)
    ax_trace.plot(steps, diffs.counts, color=COUNT_COLOR, label="changed voxels")
    ax_trace.set_xlabel("Step")
    ax_trace.set_ylabel("Changed voxels", color=COUNT_COLOR)
    ax_trace.tick_params(axis="y", labelcolor=COUNT_COLOR)
    ax_trace.set_xlim(0, max(diffs.n_steps - 1, 1))

    ax_total = ax_trace.twinx()
    ax_total.plot(steps, diffs.totals, color=TOTAL_COLOR, label="total change")
    ax_total.set_ylabel("Total |change|", color=TOTAL_COLOR)
    ax_total.tick_params(axis="y", labelcolor=TOTAL_COLOR)

    trace_cursor = ax_trace.axvline(0, color="black", linewidth=1, linestyle="--")

    def update_frame(step: int):
        image.set_data(sm.overlay(step))
        label = "Frame" if not sm.has_segmentation else "Frame + Segmentation"
        ax_image.set_title(f"{label} (Step {step}/{n_frames - 1})")

        ax_diff.clear()
        style_3d(ax_diff)
        centres, magnitudes = diffs.at(step)
        # Crop to the object-scale axis limits: 3D axes don't clip scatters.
        inside = within_limits(centres, xlim, ylim, zlim)
        centres, magnitudes = centres[inside], magnitudes[inside]
        if len(centres):
            ax_diff.scatter(
                centres[:, 0],
                centres[:, 1],
                centres[:, 2],
                c=magnitudes,
                cmap=DIFF_CMAP,
                norm=norm,
                s=marker_size,
                depthshade=False,
            )
        else:
            ax_diff.text2D(
                0.5, 0.5, "No change", transform=ax_diff.transAxes, ha="center"
            )
        total = float(magnitudes.sum()) if len(magnitudes) else 0.0
        ax_diff.set_title(
            f"Changed voxels ({len(centres)}, total {total:.2f}, "
            f"Step {step}/{n_frames - 1})"
        )

        # An axvline is a two-point Line2D, so both x values must be set.
        trace_cursor.set_xdata([step] * 2)
        ax_trace.set_title("Change over the episode")

        return [image]

    anim = FuncAnimation(
        fig, update_frame, frames=n_frames, interval=interval, blit=True
    )

    visualizations_dir = exp_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    gif_path = visualizations_dir / f"salience3d_diff_{episode}.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps))

    print(f"Animation saved to: {gif_path}")
    plt.close()

    return gif_path


def main() -> None:
    """Animate the salience3d diff of every requested episode.

    Raises:
        SystemExit: If the experiment directory holds no detailed stats.
    """
    parser = argparse.ArgumentParser(
        description="Animate the SalienceSM's per-step 3D salience change."
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
    parser.add_argument(
        "--sensor-module",
        default=None,
        help="Sensor module to read, e.g. SM_9 (default: found from the stats)",
    )
    parser.add_argument("--fps", type=int, default=2, help="GIF frames per second")
    args = parser.parse_args()

    episodes = (
        [args.episode] if args.episode is not None else available_episodes(args.exp_dir)
    )
    if not episodes:
        raise SystemExit(f"No detailed stats found under {args.exp_dir}")
    for episode in episodes:
        create_salience3d_diff_animation(
            args.exp_dir,
            episode=episode,
            sensor_module_id=args.sensor_module,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
