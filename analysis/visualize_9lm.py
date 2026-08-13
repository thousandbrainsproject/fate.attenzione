"""Animate a 9-LM experiment: camera view, patch array, and per-LM MLHs.

Three panels, step by step:

* Left: the view_finder camera frame with its segmentation tinted green
  (SM_9 in the 9-patch setup).
* Middle: the nine patch-camera views (SM_0..SM_8), arranged in the same
  3x3 spatial layout the live plotter uses, with patch_0 at the center::

      patch_1  patch_2  patch_3
      patch_4  patch_0  patch_5
      patch_6  patch_7  patch_8

* Right: each LM's most likely hypothesis as a point cloud of the learned
  object model (as in the live plotter's MLH panel), in the same 3x3 layout
  as the patch that feeds it. Points are colored by object identity so
  cross-LM agreement is visible at a glance; the red dot is the hypothesized
  sensor location on the model.

A second row holds a per-LM status table: MLH object (cell tinted with the
object's cloud color), MLH evidence, whether the LM sent a GSG goal this
step (from the attention telemetry's ``pre_filter_goals``), and whether it
sent an inhibitory region this step (from the attention telemetry's
``inhibitory_senders``; absent in runs predating it). A patch greys out on
steps where attention filtered its percept (``filtered_percepts`` in the
attention telemetry; absent in runs predating it); MLH tiles and table rows
grey out on steps where their LM did not process (``lm_processed_steps``).

The object point clouds come from the pretrained model the experiment
loaded (``--model-path``), since learned graphs are not part of the
detailed stats.

Usage::

    uv run python analysis/visualize_9lm.py [exp_dir] [--episode N]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from detailed_stats import available_episodes, extract_rgba, load_episode_stats
from visualize_3d import (
    OKABE_ITO,
    AttentionTelemetry,
    LMEvidence,
    SMTelemetry,
    _bounds,
)
from visualize_goal_filtering import GoalFilteringTelemetry, interior_points

DEFAULT_EXP_DIR = (
    Path(os.environ.get("MONTY_LOGS", "~/tbp/results/monty")).expanduser()
    / "projects"
    / "monty_runs"
    / "potato_9"
)

DEFAULT_MODEL_PATH = (
    Path(os.environ.get("MONTY_MODELS", "~/tbp/results/monty/pretrained_models"))
    .expanduser()
    / "pretrained_ycb_v13"
    / "surf_agent_1lm_10distinctobj"
    / "pretrained"
    / "model.pt"
)

# Mirrors live_plotter._NINE_PATCH_IDS: the sensors' spatial arrangement,
# with the central patch_0 at the center of the grid. LM_i sits in the same
# cell as patch_i, its input.
PATCH_GRID = ((1, 2, 3), (4, 0, 5), (6, 7, 8))

VIEW_FINDER_SM = 9

# Live plotter's dimming for non-contributing patches.
FILTERED_ALPHA = 0.45
FILTERED_GREY_BLEND = 0.65


def load_object_clouds(model_path: Path) -> dict[str, np.ndarray]:
    """Load each learned object's point cloud from a pretrained model.

    Args:
        model_path: Path to the pretrained ``model.pt``.

    Returns:
        Object name to ``(N, 3)`` model-frame point positions.
    """
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    graph_memory = state["lm_dict"][0]["graph_memory"]
    clouds = {}
    for name, channels in graph_memory.items():
        graph = next(iter(channels.values()))
        clouds[name] = np.asarray(graph.pos)
    return clouds


def object_colors(lms: dict[int, LMEvidence]) -> dict[str, str]:
    """Assign one stable color per object name across all LMs.

    Objects are ranked by their peak evidence anywhere, so the strongest
    hypotheses get the most distinguishable colors.

    Args:
        lms: Per-LM evidence records.

    Returns:
        Object name to color; objects beyond the palette share grey.
    """
    peaks: dict[str, float] = {}
    for lme in lms.values():
        for name, trace in lme.max_evidences.items():
            peaks[name] = max(peaks.get(name, -np.inf), float(trace.max()))
    ranked = sorted(peaks, key=lambda name: peaks[name], reverse=True)
    palette = OKABE_ITO + tuple(plt.cm.tab20.colors)
    return {
        name: palette[i] if i < len(palette) else "0.7"
        for i, name in enumerate(ranked)
    }


def filtered_patches_per_step(stats: dict) -> list[set[int]]:
    """Find which patches' percepts were filtered by attention on each step.

    Filtered percepts are recorded by sender id (the sensor module id, e.g.
    ``patch_4``) in the attention telemetry.

    Args:
        stats: Loaded episode stats.

    Returns:
        One set of patch indices per step; empty if the run predates the
        ``filtered_percepts`` telemetry.
    """
    per_step = []
    for filtered in stats.get("attention_system", {}).get("filtered_percepts", []):
        patches = set()
        for sender_id in filtered:
            suffix = str(sender_id).rsplit("_", 1)[-1]
            if suffix.isdigit():
                patches.add(int(suffix))
        per_step.append(patches)
    return per_step


def inhibiting_lms_per_step(stats: dict, n_frames: int) -> list[set[int]]:
    """Find which LMs sent an inhibitory region on each step.

    The attention system records the senders of negative-weight regions
    (``inhibitory_senders``, sender ids like ``learning_module_3``) once per
    ``update_regions`` call, and that runs more than once per step (once with
    SM regions, once with LM regions). Entries are therefore chunked evenly
    across frames and unioned within each chunk.

    Args:
        stats: Loaded episode stats.
        n_frames: Number of animation frames, for the chunked mapping.

    Returns:
        One set of LM indices per frame; empty if the run predates the
        ``inhibitory_senders`` telemetry.
    """
    entries = stats.get("attention_system", {}).get("inhibitory_senders", [])
    if not entries or n_frames <= 0:
        return []
    ratio = max(1, round(len(entries) / n_frames))
    per_step = []
    for frame in range(n_frames):
        senders = set()
        for filtered in entries[frame * ratio : (frame + 1) * ratio]:
            for sender_id in filtered:
                suffix = str(sender_id).rsplit("_", 1)[-1]
                if suffix.isdigit():
                    senders.add(int(suffix))
        per_step.append(senders)
    return per_step


def gsg_senders_per_step(stats: dict) -> list[set[int]]:
    """Find which LMs sent a GSG goal on each step.

    GSG goals carry ``sender_id = learning_module_id`` (e.g.
    ``learning_module_3``) and ride in the attention telemetry's pre-filter
    goals.

    Args:
        stats: Loaded episode stats.

    Returns:
        One set of LM indices per step.
    """
    per_step = []
    for goals in stats.get("attention_system", {}).get("pre_filter_goals", []):
        senders = set()
        for g in goals:
            if g.get("sender_type") != "GSG":
                continue
            suffix = str(g.get("sender_id", "")).rsplit("_", 1)[-1]
            if suffix.isdigit():
                senders.add(int(suffix))
        per_step.append(senders)
    return per_step


def grey_out(rgba: np.ndarray) -> np.ndarray:
    """Dim a patch image the way the live plotter dims filtered patches.

    Args:
        rgba: ``(H, W, 4)`` uint8 image.

    Returns:
        A float RGBA image, grey-blended and faded.
    """
    rgb = rgba[..., :3].astype(float) / 255.0
    grey = rgb.mean(axis=-1, keepdims=True)
    dimmed = (1.0 - FILTERED_GREY_BLEND) * rgb + FILTERED_GREY_BLEND * grey
    alpha = np.full((*dimmed.shape[:2], 1), FILTERED_ALPHA)
    return np.concatenate([dimmed, alpha], axis=-1)


def create_9lm_animation(
    exp_dir: Path,
    episode: int = 0,
    model_path: Path = DEFAULT_MODEL_PATH,
    fps: int = 2,
    interval: int = 500,
) -> Path:
    """Animate the camera view, patch array, and per-LM MLH point clouds.

    Args:
        exp_dir: Experiment directory.
        episode: Episode to visualize.
        model_path: Pretrained model supplying the object point clouds.
        fps: Frames per second of the saved gif.
        interval: Milliseconds between animation frames.

    Returns:
        Path to the saved gif.
    """
    stats = load_episode_stats(exp_dir, episode=episode)
    view = SMTelemetry(stats, VIEW_FINDER_SM)
    n_frames = view.n_frames

    patch_rgbas: dict[int, np.ndarray | None] = {}
    for i in range(9):
        try:
            patch_rgbas[i] = extract_rgba(stats, i)
        except KeyError:
            patch_rgbas[i] = None
    if all(r is None for r in patch_rgbas.values()):
        print(
            "No patch images in these stats - record them by setting "
            "save_raw_obs: true for sensor_module_0..8 in potato_9.yaml."
        )

    lms = {i: LMEvidence(stats, i) for i in range(9)}
    colors = object_colors(lms)
    clouds = load_object_clouds(model_path)
    mlhs = {i: stats.get(f"LM_{i}", {}).get("current_mlh", []) for i in range(9)}
    gsg_steps = gsg_senders_per_step(stats)
    filtered_steps = filtered_patches_per_step(stats)
    inhibition_steps = inhibiting_lms_per_step(stats, n_frames)
    processed = {
        i: np.asarray(stats.get(f"LM_{i}", {}).get("lm_processed_steps", []),
                      dtype=bool)
        for i in range(9)
    }

    attention = AttentionTelemetry(stats, feature="weight")

    fig = plt.figure(figsize=(19, 10))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.1], hspace=0.12)
    grid = outer[0].subgridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.15)
    ax_camera = fig.add_subplot(grid[0, 0])
    patch_grid = grid[0, 1].subgridspec(3, 3, wspace=0.05, hspace=0.08)
    mlh_grid = grid[0, 2].subgridspec(3, 3, wspace=0.05, hspace=0.08)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.15)
    ax_table = fig.add_subplot(bottom[0, 0])
    ax_table.axis("off")
    ax_voxels = fig.add_subplot(bottom[0, 1], projection="3d")

    # Voxel grid styling mirrors visualize_3d: fixed diverging scale centered
    # on zero (red = repulsion, blue = attraction). Bounds are the object
    # neighborhood as the goal scripts define it -- the interior of the goal
    # cloud, which excludes the room's surfaces -- because attended wall and
    # floor voxels accumulate across the episode and would otherwise stretch
    # the view to room scale. Off-view voxels are counted in the title.
    lifetime = float(attention.voxel_lifetime or 1)
    # Two anchors reliably mark the object neighborhood: the first frame's
    # goal-cloud interior (episodes start fixated on the primary target), and
    # the negative-weight voxels (inhibition marks recognized objects).
    anchors = []
    goal_steps = GoalFilteringTelemetry(stats).pre
    if goal_steps and len(goal_steps[0]):
        anchors.extend(interior_points([goal_steps[0]]))
    negative = [
        c[w < 0]
        for c, w in zip(attention.centres, attention.values)
        if len(c) and (w < 0).any()
    ]
    if negative:
        anchors.append(np.vstack(negative))
    if not anchors:
        anchors = [c for c in attention.centres if len(c)]
    vox_bounds = _bounds(anchors)
    vox_low = np.array([vox_bounds[0][0], vox_bounds[1][0], vox_bounds[2][0]])
    vox_high = np.array([vox_bounds[0][1], vox_bounds[1][1], vox_bounds[2][1]])

    def style_voxels() -> None:
        ax_voxels.set_xlim(vox_bounds[0])
        ax_voxels.set_ylim(vox_bounds[1])
        ax_voxels.set_zlim(vox_bounds[2])
        ax_voxels.set_box_aspect([1, 1, 1])
        # Top-down, as in visualize_3d's voxel panel.
        ax_voxels.view_init(elev=90, azim=-90)
        ax_voxels.set_xticklabels([])
        ax_voxels.set_yticklabels([])
        ax_voxels.set_zticklabels([])

    style_voxels()
    voxel_anchor = ax_voxels.scatter(
        [], [], [], c=[], cmap="RdBu", s=4, alpha=0.8,
        vmin=-lifetime, vmax=lifetime,
    )
    voxel_bar = plt.colorbar(
        voxel_anchor, ax=ax_voxels, fraction=0.046, pad=0.08
    )
    voxel_bar.set_label("voxel weight", rotation=270, labelpad=12)

    fig.suptitle("9-LM Overview", fontsize=14, fontweight="bold")

    ax_camera.axis("off")
    camera_image = ax_camera.imshow(view.overlay(0))
    height, width = view.rgbas[0].shape[:2]
    ax_camera.add_patch(
        Rectangle(
            (width // 2 - 1.5, height // 2 - 1.5),
            3,
            3,
            linewidth=1,
            edgecolor="black",
            facecolor="none",
        )
    )

    patch_axes = {}
    patch_images = {}
    mlh_axes = {}
    for row in range(3):
        for col in range(3):
            idx = PATCH_GRID[row][col]
            ax = fig.add_subplot(patch_grid[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            patch_axes[idx] = ax
            rgbas = patch_rgbas[idx]
            if rgbas is None:
                ax.text(0.5, 0.5, "not\nrecorded", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="0.5")
                patch_images[idx] = None
            else:
                patch_images[idx] = ax.imshow(rgbas[0])

            mlh_axes[idx] = fig.add_subplot(mlh_grid[row, col], projection="3d")


    def draw_mlh(idx: int, frame: int) -> None:
        ax = mlh_axes[idx]
        ax.cla()
        ax.set_axis_off()

        lme = lms[idx]
        lm_step = lme.step_for_frame(frame)
        per_step = mlhs[idx]
        if lm_step < 0 or lm_step >= len(per_step):
            ax.text2D(0.5, 0.5, "—", transform=ax.transAxes, ha="center")
            return
        mlh = per_step[lm_step]
        graph_id = mlh.get("graph_id")
        cloud = clouds.get(graph_id)
        if cloud is None:
            ax.text2D(0.5, 0.5, "—", transform=ax.transAxes, ha="center")
            return

        # Rotate the model into its estimated pose, about the model's center:
        # graphs are stored in training-world coordinates (habitat, y-up,
        # object placed ~1.5m up), so rotating about the origin would swing
        # the cloud through a huge arc. The recorded rotation is
        # BufferEncoder's as_euler("xyz", degrees=True) of the MLH rotation --
        # "how the object model needs to be rotated to be consistent with the
        # observations" -- so it applies forward to model-frame points.
        euler = mlh.get("rotation")
        pose = (
            Rotation.from_euler("xyz", np.asarray(euler, dtype=float), degrees=True)
            if euler is not None
            else Rotation.identity()
        )
        center = cloud.mean(axis=0)
        posed = pose.apply(cloud - center)

        voting = frame < len(processed[idx]) and bool(processed[idx][frame])
        color = colors.get(graph_id, "0.7") if voting else "0.75"
        alpha = 0.8 if voting else FILTERED_ALPHA
        # Native model axes. (The live plotter's (y, x, z) swap is an odd
        # permutation -- a reflection -- which mirrors any applied rotation.)
        ax.scatter(posed[:, 0], posed[:, 1], posed[:, 2], c=[color], s=1,
                   alpha=alpha)
        location = mlh.get("location")
        if location is not None:
            location = pose.apply(np.asarray(location, dtype=float) - center)
            ax.scatter([location[0]], [location[1]], [location[2]],
                       c="red", s=15, alpha=1.0 if voting else FILTERED_ALPHA)
        ax.set_aspect("equal")
        # The model frame is y-up (habitat training convention), so display
        # with y as screen-up, viewed from the training camera's side: an
        # identity pose renders the object upright exactly as it was learned.
        ax.view_init(elev=0, azim=-90, vertical_axis="y")

    def draw_table(step: int, senders: set[int]) -> None:
        """Rebuild the per-LM status table for one step."""
        ax_table.clear()
        ax_table.axis("off")
        cells = []
        cell_colors = []
        for i in range(9):
            lme = lms[i]
            lm_step = lme.step_for_frame(step)
            per_step = mlhs[i]
            if 0 <= lm_step < len(per_step):
                mlh = per_step[lm_step]
                graph_id = str(mlh.get("graph_id", "—"))
                evidence = f"{float(mlh.get('evidence', 0.0)):.2f}"
            else:
                graph_id, evidence = "—", "—"
            inhibiting = (
                step < len(inhibition_steps) and i in inhibition_steps[step]
            )
            cells.append(
                [
                    graph_id,
                    evidence,
                    "✓" if i in senders else "",
                    "✓" if inhibiting else "",
                ]
            )
            object_color = (
                mcolors.to_rgba(colors[graph_id], alpha=0.45)
                if graph_id in colors
                else "white"
            )
            cell_colors.append(
                [
                    object_color,
                    "white",
                    "white",
                    mcolors.to_rgba("firebrick", alpha=0.3)
                    if inhibiting
                    else "white",
                ]
            )
        table = ax_table.table(
            cellText=cells,
            cellColours=cell_colors,
            rowLabels=[f"LM_{i}" for i in range(9)],
            colLabels=("MLH object", "MLH evidence", "goal sent", "inhibition"),
            cellLoc="center",
            loc="center",
            colWidths=(0.28, 0.16, 0.12, 0.12),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.15)
        # Grey the rows of LMs that did not process this step.
        for i in range(9):
            voting = step < len(processed[i]) and bool(processed[i][step])
            if not voting:
                for col in range(4):
                    table[i + 1, col].get_text().set_color("0.6")

    def update_frame(step: int):
        camera_image.set_data(view.overlay(step))
        label = "Frame" if not view.has_segmentation else "Frame + Segmentation"
        ax_camera.set_title(f"{label} (Step {step}/{n_frames - 1})")

        # A patch greys out when attention filtered its percept this step.
        filtered = filtered_steps[step] if step < len(filtered_steps) else set()
        for idx, image in patch_images.items():
            rgbas = patch_rgbas[idx]
            if image is None or step >= len(rgbas):
                continue
            image.set_data(
                grey_out(rgbas[step]) if idx in filtered else rgbas[step]
            )

        senders = gsg_steps[step] if step < len(gsg_steps) else set()
        for idx in mlh_axes:
            draw_mlh(idx, step)
        draw_table(step, senders)

        ax_voxels.clear()
        style_voxels()
        centres, weights = attention.voxels_at(step)
        if len(centres):
            # Crop to the robust bounds: 3D axes don't clip scatters, so
            # far-flung inhibition voxels would bleed across the figure.
            in_view = ((centres >= vox_low) & (centres <= vox_high)).all(axis=1)
            shown, shown_weights = centres[in_view], weights[in_view]
            if len(shown):
                ax_voxels.scatter(
                    shown[:, 0], shown[:, 1], shown[:, 2],
                    c=shown_weights, cmap="RdBu", s=6, alpha=0.8,
                    vmin=-lifetime, vmax=lifetime,
                )
            ax_voxels.set_title(
                f"Attention grid ({len(shown)}/{len(centres)} voxels in view)",
                fontsize=9,
            )
        else:
            ax_voxels.set_title("Attention grid (empty)", fontsize=9)
        return []

    anim = FuncAnimation(
        fig, update_frame, frames=n_frames, interval=interval, blit=False
    )

    visualizations_dir = exp_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    gif_path = visualizations_dir / f"9lm_{episode}.gif"
    anim.save(gif_path, writer=PillowWriter(fps=fps))

    print(f"Animation saved to: {gif_path}")
    plt.close()

    return gif_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate a 9-LM run: camera, patch array, and per-LM MLHs."
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
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Pretrained model.pt supplying the object point clouds",
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
        create_9lm_animation(
            args.exp_dir, episode=episode, model_path=args.model_path, fps=args.fps
        )


if __name__ == "__main__":
    main()
