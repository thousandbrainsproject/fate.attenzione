"""Interactively explore one step's attention goal filtering.

Opens the same two-panel view as ``visualize_goal_filtering.py`` -- the sensor
frame with its segmentation tinted green, beside a 3D scatter of pre-filter
goals (red) under post-filter goals (black) -- but as a live matplotlib window
for a single step (the first, by default), so the 3D panel can be rotated and
zoomed with the mouse.

Usage::

    uv run python analysis/interactive_goal_filtering.py [exp_dir] [--step N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from detailed_stats import load_episode_stats
from visualize_goal_filtering import (
    PRE_COLOR,
    POST_COLOR,
    clip_to_limits,
    interior_points,
)
from visualize_3d import DEFAULT_EXP_DIR, _bounds


def load_step(
    stats: dict, step: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Pull one step's frame, segmentation, and goal sets out of episode stats.

    Args:
        stats: Loaded episode stats.
        step: Which step to read.

    Returns:
        An ``(rgba, segmentation_mask, pre, post)`` tuple; the mask is None if
        no segmentation was recorded and the goal arrays are ``(N, 3)``.

    Raises:
        IndexError: If the step is beyond the recorded frames.
    """
    sm = stats["SM_1"]
    raw = sm["raw_observations"]
    if step >= len(raw):
        raise IndexError(f"step {step} not recorded (episode has {len(raw)} frames)")
    rgba = np.array(raw[step]["rgba"], dtype=np.uint8)

    masks = sm.get("segmentation_maps", [])
    mask = None
    if step < len(masks) and masks[step] is not None:
        mask = np.asarray(masks[step])

    attention = stats.get("attention_system", {})

    def locations(key: str) -> np.ndarray:
        per_step = attention.get(key, [])
        goals = per_step[step] if step < len(per_step) else []
        return np.array(
            [g["location"] for g in goals if g.get("location") is not None],
            dtype=float,
        ).reshape(-1, 3)

    return rgba, mask, locations("pre_filter_goals"), locations("post_filter_goals")


def tinted(rgba: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Return the frame with the segmentation mask tinted green.

    Args:
        rgba: The ``(H, W, 4)`` frame.
        mask: The segmentation mask, or None.

    Returns:
        The blended frame.
    """
    out = rgba.copy()
    if mask is None or not np.any(mask > 0):
        return out
    tint = np.zeros_like(out)
    tint[..., 1] = 255
    if out.shape[-1] == 4:
        tint[..., 3] = 128
    active = mask > 0
    out[active] = (out[active] * 0.6 + tint[active] * 0.4).astype(np.uint8)
    return out


def build_figure(stats: dict, step: int) -> plt.Figure:
    """Build the interactive two-panel figure for one step.

    Args:
        stats: Loaded episode stats.
        step: Which step to show.

    Returns:
        The assembled figure.
    """
    rgba, mask, pre, post = load_step(stats, step)

    fig = plt.figure(figsize=(13, 6.5))
    grid = fig.add_gridspec(1, 2, wspace=0.25)
    ax_image = fig.add_subplot(grid[0, 0])
    ax_goals = fig.add_subplot(grid[0, 1], projection="3d")
    fig.suptitle(
        f"Attention Goal Filtering — step {step} (drag to rotate, scroll to zoom)",
        fontsize=13,
        fontweight="bold",
    )

    ax_image.imshow(tinted(rgba, mask))
    ax_image.axis("off")
    ax_image.set_title("Frame + Segmentation" if mask is not None else "Frame")
    height, width = rgba.shape[:2]
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

    xlim, ylim, zlim = _bounds(interior_points([p for p in (pre, post) if len(p)]))
    # Crop to the object-scale axis limits: 3D axes don't clip scatters, so
    # room-surface points would bleed past the axes box. Legend keeps the
    # uncropped counts (the data), the panel shows the in-view subset.
    pre_view = clip_to_limits(pre, xlim, ylim, zlim)
    post_view = clip_to_limits(post, xlim, ylim, zlim)

    # Post-filter goals sit at the same locations as their pre counterparts;
    # draw in call order so black lands on top rather than being depth-sorted
    # under the larger red set.
    ax_goals.computed_zorder = False
    if len(pre_view):
        ax_goals.scatter(
            pre_view[:, 0], pre_view[:, 1], pre_view[:, 2],
            c=PRE_COLOR, s=6, alpha=0.5, label=f"pre-filter ({len(pre)})",
        )
    if len(post_view):
        ax_goals.scatter(
            post_view[:, 0], post_view[:, 1], post_view[:, 2],
            c=POST_COLOR, s=6, alpha=0.8, label=f"post-filter ({len(post)})",
        )
    ax_goals.set_xlim(xlim)
    ax_goals.set_ylim(ylim)
    ax_goals.set_zlim(zlim)
    ax_goals.set_box_aspect([1, 1, 1])
    ax_goals.set_xlabel("X")
    ax_goals.set_ylabel("Y")
    ax_goals.set_zlabel("Z")
    # Start top-down like the gif; from here the mouse takes over.
    ax_goals.view_init(elev=90, azim=-90)
    if len(pre) or len(post):
        ax_goals.legend(loc="upper right", fontsize=9)
    else:
        ax_goals.text2D(
            0.5, 0.5, "No goals recorded", transform=ax_goals.transAxes, ha="center"
        )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively explore one step's attention goal filtering."
    )
    parser.add_argument(
        "exp_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_EXP_DIR,
        help=f"Experiment output directory (default: {DEFAULT_EXP_DIR})",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode to load")
    parser.add_argument("--step", type=int, default=0, help="Step to show (default 0)")
    args = parser.parse_args()

    print(f"Loading episode {args.episode} from {args.exp_dir} ...")
    stats = load_episode_stats(args.exp_dir, episode=args.episode)
    build_figure(stats, args.step)
    plt.show()


if __name__ == "__main__":
    main()
