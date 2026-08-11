"""Animate how the AttentionSystem re-weights the goals it receives.

Each step, the attention system receives every goal the sensor and learning
modules proposed and returns them with confidences modulated by the voxel grid:
scaled down in repulsive (negative-weight) voxels, up in attractive ones, and
unchanged outside the grid (see ``AttentionSystem.step``). Both sides ride in
the detailed stats under::

    stats["attention_system"] = {
        ...,
        "pre_filter_goals":  [[{"location": (3,), "confidence": float, ...}, ...], ...],
        "post_filter_goals": [[{"location": (3,), "confidence": float, ...}, ...], ...],
    }

The figure pairs the sensor view (with its segmentation tinted green) with a
3D scatter of the post-weighting goals colored by their modulated confidence —
a recognized, inhibited object shows up as its goals' confidences collapsing
toward zero while the rest of the scene keeps its salience.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from detailed_stats import available_episodes, load_episode_stats
from visualize_3d import DEFAULT_EXP_DIR, SMTelemetry, _bounds


class GoalWeightingTelemetry:
    """Plot-ready views over the attention system's recorded goal weighting.

    Attributes:
        pre: Per step, ``(locations, confidences)`` of the goals handed to the
            attention system.
        post: Per step, ``(locations, confidences)`` of the goals after their
            confidences were modulated by the voxel grid.
    """

    def __init__(self, stats: dict) -> None:
        """Read the goal weighting telemetry out of loaded episode stats.

        Goals without a location have nothing to scatter, so they are dropped
        here.

        Args:
            stats: Loaded episode stats.
        """
        attention = stats.get("attention_system", {})
        self.pre = self._goals(attention.get("pre_filter_goals", []))
        self.post = self._goals(attention.get("post_filter_goals", []))

    @staticmethod
    def _goals(
        per_step_goals: list[list[dict]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        steps = []
        for goals in per_step_goals:
            located = [g for g in goals if g.get("location") is not None]
            steps.append(
                (
                    np.array(
                        [g["location"] for g in located], dtype=float
                    ).reshape(-1, 3),
                    np.array([g["confidence"] for g in located], dtype=float),
                )
            )
        return steps

    @property
    def has_goals(self) -> bool:
        """Whether any goal weighting was recorded."""
        return any(len(locations) for locations, _ in self.pre)

    def at(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the post-weighting goal locations and confidences of a step.

        Args:
            step: Which step to read.

        Returns:
            A ``(locations, confidences)`` pair, empty past the record.
        """
        if step < len(self.post):
            return self.post[step]
        return np.empty((0, 3)), np.empty(0)

    def bounds_points(self) -> list[np.ndarray]:
        """Return every goal location the episode touches, for axis limits.

        Returns:
            Point arrays spanning the incoming goals.
        """
        return [locations for locations, _ in self.pre if len(locations)]


def create_goal_weighting_animation(
    exp_dir: Path,
    episode: int = 0,
    sensor_module_id: str | int = 1,
    fps: int = 2,
    interval: int = 500,
    marker_size: int = 5,
) -> Path:
    """Animate the sensor view beside the re-weighted goals.

    Args:
        exp_dir: Experiment directory.
        episode: Episode to visualize.
        sensor_module_id: Sensor module to read; SM_1 is the SalienceSM.
        fps: Frames per second of the saved gif.
        interval: Milliseconds between animation frames.
        marker_size: Scatter marker size in the 3D panel.

    Returns:
        Path to the saved gif.
    """
    stats = load_episode_stats(exp_dir, episode=episode)
    sm = SMTelemetry(stats, sensor_module_id)
    weighting = GoalWeightingTelemetry(stats)

    if not weighting.has_goals:
        print(
            "No goal weighting telemetry in this episode - re-run the "
            "experiment with the updated AttentionSystemTelemetry."
        )

    n_frames = sm.n_frames
    xlim, ylim, zlim = _bounds(weighting.bounds_points())

    fig = plt.figure(figsize=(13, 5.5))
    grid = fig.add_gridspec(1, 2, wspace=0.25)
    ax_image = fig.add_subplot(grid[0, 0])
    ax_goals = fig.add_subplot(grid[0, 1], projection="3d")

    fig.suptitle("Attention Goal Weighting", fontsize=14, fontweight="bold")

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
    style_3d(ax_goals)

    image = ax_image.imshow(sm.overlay(0))

    # The anchor scatter exists only to carry the colorbar; update_frame clears
    # and redraws the 3D panel each step.
    anchor = ax_goals.scatter(
        [], [], [], c=[], cmap="plasma", s=marker_size, alpha=0.8, vmin=0, vmax=1
    )
    bar = plt.colorbar(anchor, ax=ax_goals, fraction=0.046, pad=0.08)
    bar.set_label("Weighted confidence", rotation=270, labelpad=15)

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

    def update_frame(step: int):
        image.set_data(sm.overlay(step))
        label = "Frame" if not sm.has_segmentation else "Frame + Segmentation"
        ax_image.set_title(f"{label} (Step {step}/{n_frames - 1})")

        ax_goals.clear()
        style_3d(ax_goals)
        locations, confidences = weighting.at(step)
        if len(locations):
            ax_goals.scatter(
                locations[:, 0],
                locations[:, 1],
                locations[:, 2],
                c=confidences,
                cmap="plasma",
                s=marker_size,
                alpha=0.8,
                vmin=0,
                vmax=1,
            )
            ax_goals.set_title(
                f"Goals by weighted confidence ({len(locations)} goals, "
                f"Step {step}/{n_frames - 1})"
            )
        else:
            ax_goals.set_title(f"Goals (none, Step {step}/{n_frames - 1})")
            ax_goals.text2D(
                0.5, 0.5, "No goals", transform=ax_goals.transAxes, ha="center"
            )

        return [image]

    anim = FuncAnimation(fig, update_frame, frames=n_frames, interval=interval,
                         blit=True)

    visualizations_dir = exp_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    gif_path = visualizations_dir / f"goal_weighting_{episode}.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps))

    print(f"Animation saved to: {gif_path}")
    plt.close()

    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate how the attention system re-weights goals."
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
        create_goal_weighting_animation(args.exp_dir, episode=episode, fps=args.fps)


if __name__ == "__main__":
    main()
