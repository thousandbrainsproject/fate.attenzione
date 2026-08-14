"""Animate the attention goal filtering and the 3D salience change together.

Each step the attention system filters the goals it was handed against its
voxel grid (``AttentionSystem.step``) while the SalienceSM diffs the salience
it just voxelized against the previous step's (``SalienceSM._step_salience3d``).
``visualize_goal_filtering`` and ``visualize_salience_diff`` animate one each,
against the same sensor view; both describe the same moment, so this puts them
in one figure::

    | frame + segmentation | goals in and out of the filter | changed voxels |

The telemetry each panel reads, and how to record it, is documented in the two
single-panel scripts. The sensor module is the one that recorded the salience3d
diff, so all three panels describe the same module's view.

The 3D panels share one set of axis limits spanning both point clouds, so a
spot in the goals panel is the same spot in the change panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from detailed_stats import available_episodes, load_episode_stats
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from visualize_3d import DEFAULT_EXP_DIR, SMTelemetry, _bounds
from visualize_goal_filtering import (
    POST_COLOR,
    PRE_COLOR,
    GoalFilteringTelemetry,
    clip_to_limits,
    interior_points,
)
from visualize_salience_diff import (
    DIFF_CMAP,
    Salience3dDiffTelemetry,
    find_salience_sm,
    within_limits,
)


def create_goals_and_diff_animation(
    exp_dir: Path,
    episode: int = 0,
    sensor_module_id: str | int | None = None,
    fps: int = 2,
    interval: int = 500,
    goal_marker_size: int = 5,
    diff_marker_size: int = 12,
) -> Path:
    """Animate the sensor view, the goal filtering, and the salience3d diff.

    Args:
        exp_dir: Experiment directory.
        episode: Episode to visualize.
        sensor_module_id: Sensor module to read; found from the stats if None.
        fps: Frames per second of the saved gif.
        interval: Milliseconds between animation frames.
        goal_marker_size: Scatter marker size in the goals panel.
        diff_marker_size: Scatter marker size in the change panel.

    Returns:
        Path to the saved gif.
    """
    stats = load_episode_stats(exp_dir, episode=episode)

    if sensor_module_id is None:
        sensor_module_id = find_salience_sm(stats)
    elif isinstance(sensor_module_id, int) or str(sensor_module_id).isdigit():
        sensor_module_id = f"SM_{sensor_module_id}"

    sm = SMTelemetry(stats, sensor_module_id)
    goals = GoalFilteringTelemetry(stats)
    diffs = Salience3dDiffTelemetry(stats, sensor_module_id)

    if not goals.has_goals:
        print(
            "No goal filtering telemetry in this episode - re-run the "
            "experiment with the updated AttentionSystemTelemetry."
        )
    if not diffs.has_diff:
        print(
            "No voxel changed by more than the module's threshold "
            f"({diffs.threshold}) in this episode."
        )

    # The diff is recorded once per non-exploring step, the frames once per
    # recorded step; take the shorter so no panel runs past its record.
    n_frames = min(sm.n_frames, diffs.n_steps)
    xlim, ylim, zlim = _bounds(
        interior_points(goals.bounds_points() + diffs.bounds_points())
    )
    norm = Normalize(*diffs.value_range())

    fig = plt.figure(figsize=(17, 5.5))
    # The change panel carries the colour bar, so it gets the extra width.
    grid = fig.add_gridspec(1, 3, wspace=0.3, width_ratios=(1, 1, 1.25))
    ax_image = fig.add_subplot(grid[0, 0])
    ax_goals = fig.add_subplot(grid[0, 1], projection="3d")
    ax_diff = fig.add_subplot(grid[0, 2], projection="3d")

    # All three panels show the same step, so the step counter lives in the
    # figure title rather than being repeated in each panel's.
    suptitle = fig.suptitle("", fontsize=14, fontweight="bold")
    # Along the bottom: a caption up by the suptitle collides with it.
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

    def style_goals(ax) -> None:
        """Style the goals panel. Re-applied after every ax.clear()."""
        # Post-filter goals sit at exactly the same locations as their
        # pre-filter counterparts, and Axes3D redraws collections sorted by
        # projected depth -- which buries the black points under the larger red
        # set. Draw in call order instead so post lands on top.
        ax.computed_zorder = False
        style_3d(ax)

    ax_image.axis("off")
    style_goals(ax_goals)
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

    def draw_goals(step: int) -> None:
        """Redraw the goals panel for one step."""
        ax_goals.clear()
        style_goals(ax_goals)
        pre, post = goals.at(step)
        # Crop to the object-scale axis limits: 3D axes don't clip scatters.
        pre = clip_to_limits(pre, xlim, ylim, zlim)
        post = clip_to_limits(post, xlim, ylim, zlim)
        # Pre first so the surviving goals draw on top of the red field.
        if len(pre):
            ax_goals.scatter(
                pre[:, 0],
                pre[:, 1],
                pre[:, 2],
                c=PRE_COLOR,
                s=goal_marker_size,
                alpha=0.5,
                label="pre-filter",
            )
        if len(post):
            ax_goals.scatter(
                post[:, 0],
                post[:, 1],
                post[:, 2],
                c=POST_COLOR,
                s=goal_marker_size,
                alpha=0.8,
                label="post-filter",
            )
        ax_goals.set_title(f"Goals ({len(pre)} pre, {len(post)} post)")
        if len(pre) or len(post):
            ax_goals.legend(loc="upper right", fontsize=8)
        else:
            ax_goals.text2D(
                0.5, 0.5, "No goals", transform=ax_goals.transAxes, ha="center"
            )

    def draw_diff(step: int) -> None:
        """Redraw the salience change panel for one step."""
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
                s=diff_marker_size,
                depthshade=False,
            )
        else:
            ax_diff.text2D(
                0.5, 0.5, "No change", transform=ax_diff.transAxes, ha="center"
            )
        total = float(magnitudes.sum()) if len(magnitudes) else 0.0
        ax_diff.set_title(f"Changed voxels ({len(centres)}, total {total:.2f})")

    def update_frame(step: int):
        image.set_data(sm.overlay(step))
        ax_image.set_title("Frame + Segmentation" if sm.has_segmentation else "Frame")
        draw_goals(step)
        draw_diff(step)
        suptitle.set_text(
            f"Attention Goals and 3D Salience Change (Step {step}/{n_frames - 1})"
        )
        return [image]

    anim = FuncAnimation(
        fig, update_frame, frames=n_frames, interval=interval, blit=True
    )

    visualizations_dir = exp_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    gif_path = visualizations_dir / f"goals_and_salience_diff_{episode}.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps))

    print(f"Animation saved to: {gif_path}")
    plt.close()

    return gif_path


def main() -> None:
    """Animate the goals and the salience3d diff of every requested episode.

    Raises:
        SystemExit: If the experiment directory holds no detailed stats.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Animate the attention goal filtering beside the SalienceSM's "
            "per-step 3D salience change."
        )
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
        create_goals_and_diff_animation(
            args.exp_dir,
            episode=episode,
            sensor_module_id=args.sensor_module,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
