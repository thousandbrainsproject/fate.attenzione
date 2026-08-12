# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Monty, Observations
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.plot_utils import add_patch_outline_to_view_finder

# turn interactive plotting off -- call plt.show() to open all figures
plt.ioff()

# Numbered patch sensors used by multi-patch experiments (e.g. potato_9).
# Ordered for a 3x3 grid matching sensor spatial layout, with patch_0 at center:
#   patch_1  patch_2  patch_3
#   patch_4  patch_0  patch_5
#   patch_6  patch_7  patch_8
_NINE_PATCH_IDS = tuple(SensorID(f"patch_{i}") for i in (1, 2, 3, 4, 0, 5, 6, 7, 8))

# Voxel coordinates are lower corners; offset to centres for plotting.
_VOXEL_CENTRE_OFFSET = 0.5
# How filtered patch images are dimmed in the live plotter.
_FILTERED_ALPHA = 0.45
_FILTERED_GREY_BLEND = 0.65


class LivePlotter:
    """Class for plotting sensor observations during an experiment.

    Set the `show_sensor_output` flag in the experiment config to True to enable live
    plotting.

    WARNING: This plotter makes a number of assumptions right now. For example, it
    assumes that
    - sensor with ID "view_finder" exists
    - sensor with ID "patch" or "patch_0".."patch_8" exists
    - "rgba" modality in "view_finder" sensor observation
    - "depth" modality in the first patch sensor observation (single-patch mode)
    - "rgba" modality in "patch_0".."patch_8" (nine-patch mode)
    - optional SalienceSM view-finder segmentation telemetry when
      ``save_segmentation`` is enabled
    """

    def __init__(self):
        self._nine_patch_mode = False
        self.save_dir: Path | None = None
        # Inhibit points held on the LM after terminal checks are only applied by
        # the attention system on the *next* Monty step. Buffer them so the plot
        # shows inhibit on the step it actually affected the grid.
        self._inhibit_applied_locations: np.ndarray = np.empty((0, 3))
        # Growing axis limits so the attention panel does not jump/zoom each step.
        self._attention_bounds: tuple[np.ndarray, np.ndarray] | None = None

    def initialize_online_plotting(
        self,
        model: Monty | None = None,
        save_dir: Path | str | None = None,
    ):
        """Create the live-plotting figure.

        Args:
            model: When provided, used to detect a nine-patch layout
                (``patch_0`` … ``patch_8``). If those sensors are present, the
                middle panel becomes a 3x3 grid of RGB patch images instead of a
                single depth image.
            save_dir: Optional directory for per-step PNG frames. Created if it
                does not exist.
        """
        self._nine_patch_mode = self._has_nine_patches(model)

        # Build mixed 2D/3D axes explicitly. plt.subplots() leaves an orphaned 2D
        # axis behind if we later replace a slot with projection="3d".
        # Layout: camera | sensor/patches | MLH | attention side
        #                                    (empty below) | attention top
        if self._nine_patch_mode:
            self.fig = plt.figure(figsize=(14, 8))
            self.fig.subplots_adjust(top=0.9, wspace=0.2, hspace=0.25)
            gs = gridspec.GridSpec(
                2,
                4,
                figure=self.fig,
                width_ratios=[1.1, 0.65, 1.0, 1.0],
                height_ratios=[1.0, 0.85],
            )
            camera_ax = self.fig.add_subplot(gs[0, 0])
            # Compact 3x3 nested in the sensor column of the top row.
            patch_gs = gs[0, 1].subgridspec(3, 3, wspace=0.05, hspace=0.2)
            self.patch_axes = [
                self.fig.add_subplot(patch_gs[row, col])
                for row in range(3)
                for col in range(3)
            ]
            mlh_ax = self.fig.add_subplot(gs[0, 2], projection="3d")
            attn_side_ax = self.fig.add_subplot(gs[0, 3], projection="3d")
            attn_top_ax = self.fig.add_subplot(gs[1, 3], projection="3d")
            # Keep self.ax[0]/[2] as camera / MLH; middle slot unused in 9-patch mode.
            self.ax = [camera_ax, None, mlh_ax, attn_side_ax, attn_top_ax]
        else:
            self.fig = plt.figure(figsize=(12, 9))
            self.fig.subplots_adjust(top=1.05, hspace=0.3, wspace=0.2)
            gs = self.fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85])
            self.ax = [
                self.fig.add_subplot(gs[0, 0]),
                self.fig.add_subplot(gs[0, 1]),
                self.fig.add_subplot(gs[0, 2], projection="3d"),
                self.fig.add_subplot(gs[0, 3], projection="3d"),
                self.fig.add_subplot(gs[1, 3], projection="3d"),
            ]
            self.patch_axes = []

        self.setup_camera_ax()
        self.setup_sensor_ax()
        self.setup_mlh_ax()
        self.setup_attention_ax()

        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self._inhibit_applied_locations = np.empty((0, 3))
        self._attention_bounds = None

    @staticmethod
    def _has_nine_patches(model: Monty | None) -> bool:
        if model is None:
            return False
        sensor_ids = {sm.sensor_module_id for sm in model.sensor_modules}
        return all(pid in sensor_ids for pid in _NINE_PATCH_IDS)

    def hardcoded_assumptions(self, observation: Observations, model: Monty):
        """Extract some of the hardcoded assumptions from the observation.

        TODO: Don't do this. It is here for now to highlight the fragility of the
        live plotter implementation at the call site. We should make this less
        fragile by passing the necessary information to the live plotter.

        Args:
            observation: The observation from the environment interface.
            model: The model.

        Returns:
            A tuple of the first learning module, the first sensor module raw
            observations, the patch depth, the view finder rgba, mlh, mlh model,
            optionally a list of RGB images for patch_0..patch_8, the
            attention system (or None), all learning modules, the set of
            patch sensor IDs whose percepts were filtered by attention, and
            the latest view-finder segmentation mask (or None).
        """
        first_learning_module = model.learning_modules[0]
        first_sensor_module = model.sensor_modules[0]
        first_sensor_module_raw_observations = (
            first_sensor_module._snapshot_telemetry.raw_observations
        )
        first_sensor_module_id = first_sensor_module.sensor_module_id

        # Find agent_id corresponding to the first_sensor_module_id
        first_sensor_module_agent_id: AgentID | None = None
        for agent_id, agent_observations in observation.items():
            if first_sensor_module_id in agent_observations:
                first_sensor_module_agent_id = agent_id
                break
        assert first_sensor_module_agent_id is not None

        agent_obs = observation[first_sensor_module_agent_id]
        first_sensor_depth = agent_obs[first_sensor_module_id]["depth"]
        view_finder_rgba = agent_obs[SensorID("view_finder")]["rgba"]

        patch_rgbs = None
        if self._nine_patch_mode:
            patch_rgbs = [
                np.asarray(agent_obs[pid]["rgba"])[..., :3] for pid in _NINE_PATCH_IDS
            ]

        if hasattr(first_learning_module, "get_current_mlh"):
            mlh = first_learning_module.get_current_mlh()
            if mlh["graph_id"] == "no_observations_yet":
                mlh_model = None
            else:
                mlh_model = first_learning_module.graph_memory.get_graph(
                    mlh["graph_id"]
                )[first_sensor_module_id]
        else:
            mlh = None
            mlh_model = None
        attention_system = getattr(model, "attention_system", None)
        filtered_patch_ids = self._attention_filtered_patch_ids(
            getattr(model, "sensor_module_outputs", None),
            attention_system,
        )
        segmentation_mask = self._latest_view_finder_segmentation(model)
        return (
            first_learning_module,
            first_sensor_module_raw_observations,
            first_sensor_depth,
            view_finder_rgba,
            mlh,
            mlh_model,
            patch_rgbs,
            attention_system,
            list(model.learning_modules),
            filtered_patch_ids,
            segmentation_mask,
        )

    @staticmethod
    def _latest_view_finder_segmentation(model: Monty) -> np.ndarray | None:
        """Return the most recent view-finder segmentation mask, if any.

        Args:
            model: Monty model whose sensor modules may include a SalienceSM
                view finder with ``save_segmentation`` telemetry.

        Returns:
            The latest ``(H, W)`` mask, or None when unavailable.
        """
        for sm in model.sensor_modules:
            if sm.sensor_module_id != SensorID("view_finder"):
                continue
            telemetry = getattr(sm, "_snapshot_telemetry", None)
            maps = getattr(telemetry, "segmentation_maps", None)
            if not maps:
                return None
            mask = maps[-1]
            return None if mask is None else np.asarray(mask)
        return None

    def _attention_filtered_patch_ids(
        self,
        sensor_module_outputs,
        attention_system,
    ) -> frozenset[SensorID]:
        """Return patch sensor IDs whose percepts were filtered by attention.

        A percept is filtered when the attention voxel grid is non-empty and the
        percept location falls outside that grid (see
        ``AttentionSystem.filter_percepts``).

        Args:
            sensor_module_outputs: Percepts after ``filter_percepts``, or None.
            attention_system: The model's attention system, or None.

        Returns:
            Frozen set of filtered patch ``SensorID``s.
        """
        if (
            attention_system is None
            or len(attention_system.grid) == 0
            or not sensor_module_outputs
        ):
            return frozenset()

        patch_ids = (
            frozenset(_NINE_PATCH_IDS)
            if self._nine_patch_mode
            else frozenset({SensorID("patch"), SensorID("patch_0")})
        )
        candidates: list[tuple[SensorID, np.ndarray]] = []
        for percept in sensor_module_outputs:
            if percept is None or percept.location is None:
                continue
            sender_id = SensorID(percept.sender_id)
            if sender_id not in patch_ids:
                continue
            candidates.append((sender_id, np.asarray(percept.location, dtype=float)))
        if not candidates:
            return frozenset()

        contained = attention_system.contains_points(
            np.stack([loc for _, loc in candidates])
        )
        return frozenset(
            sender_id
            for (sender_id, _), keep in zip(candidates, contained)
            if not keep
        )

    def show_observations(
        self,
        first_learning_module,
        first_sensor_module_raw_observations,
        first_sensor_depth,
        view_finder_rgba,
        mlh,
        mlh_model,
        patch_rgbs,
        attention_system,
        learning_modules,
        filtered_patch_ids,
        segmentation_mask,
        step: int,
        is_saccade_on_image_data_loader=False,
    ) -> None:
        self.show_view_finder(
            first_sensor_module_raw_observations,
            first_learning_module,
            first_sensor_depth,
            view_finder_rgba,
            is_saccade_on_image_data_loader,
            segmentation_mask=segmentation_mask,
        )
        if patch_rgbs is not None:
            self.show_patches(patch_rgbs, filtered_patch_ids)
        else:
            self.show_patch(first_sensor_depth, filtered_patch_ids)
        if mlh_model:
            self.show_mlh(mlh, mlh_model)
        # propose_region after this step is what will be applied next step. What
        # attention already applied this step is the buffered previous proposal.
        pending_inhibit = self._inhibited_locations(learning_modules)
        applied_inhibit = self._inhibit_applied_locations
        self._inhibit_applied_locations = pending_inhibit
        status = self._recognition_status(
            first_learning_module, inhibit_applied=len(applied_inhibit) > 0
        )
        self.fig.suptitle(f"Observation at step {step}")
        self.show_attention(
            attention_system,
            status=status,
            inhibited_locations=applied_inhibit,
        )
        plt.pause(0.00001)
        self._save_frame(step)

    def _save_frame(self, step: int) -> None:
        """Write the current figure to ``save_dir`` if one was configured.

        Args:
            step: Episode step index used in the filename.
        """
        if self.save_dir is None:
            return
        path = self.save_dir / f"step_{step:04d}.png"
        self.fig.savefig(path, dpi=120, bbox_inches="tight")

    def _inhibited_locations(self, learning_modules) -> np.ndarray:
        """Locations from all LMs' current negative attention regions.

        Args:
            learning_modules: Learning modules to query via ``propose_region``.

        Returns:
            An ``(N, 3)`` array of inhibited locations, or empty when none.
        """
        locations = []
        for lm in learning_modules:
            if not hasattr(lm, "propose_region"):
                continue
            region = lm.propose_region() or []
            locations.extend(
                aw.location
                for aw in region
                if aw.location is not None and aw.weight < 0
            )
        if not locations:
            return np.empty((0, 3))
        return np.asarray(locations, dtype=float)

    def _recognition_status(
        self, first_learning_module, inhibit_applied: bool = False
    ) -> str:
        """Summarize terminal state and applied object-region inhibition.

        Args:
            first_learning_module: The LM used for terminal-state text.
            inhibit_applied: Whether inhibit points were applied to attention
                this step (lagged one step from ``propose_region``).

        Returns:
            A short status string, or empty when neither applies.
        """
        parts = []
        terminal = getattr(first_learning_module, "terminal_state", None)
        if terminal is not None:
            parts.append(f"terminal={terminal}")
        if inhibit_applied:
            parts.append("inhibit applied")
        return " | ".join(parts)

    def show_view_finder(
        self,
        first_sensor_module_raw_observations,
        first_learning_module,
        first_sensor_depth,
        view_finder_rgba,
        is_saccade_on_image_data_loader,
        segmentation_mask=None,
    ):
        if self.camera_image:
            self.camera_image.remove()

        view_finder_rgba = self._overlay_segmentation(
            view_finder_rgba, segmentation_mask
        )

        if is_saccade_on_image_data_loader:
            center_pixel_id = np.array([200, 200])
            patch_size = np.array(first_sensor_depth).shape[0]
            raw_obs = first_sensor_module_raw_observations
            if len(raw_obs) > 0:
                center_pixel_id = np.array(raw_obs[-1]["pixel_loc"])
                view_finder_rgba = add_patch_outline_to_view_finder(
                    view_finder_rgba, center_pixel_id, patch_size
                )
            self.camera_image = self.ax[0].imshow(view_finder_rgba, zorder=-99)
        else:
            self.camera_image = self.ax[0].imshow(
                view_finder_rgba,
                zorder=-99,
            )
            # Show a square in the middle as a rough estimate of where the patch is
            # Note: This isn't exactly the size that the patch actually is.
            image_shape = view_finder_rgba.shape
            square = plt.Rectangle(
                (image_shape[1] * 4.5 // 10, image_shape[0] * 4.5 // 10),
                image_shape[1] / 10,
                image_shape[0] / 10,
                fc="none",
                ec="white",
            )
            self.ax[0].add_patch(square)
        if hasattr(first_learning_module, "get_current_mlh"):
            mlh = first_learning_module.get_current_mlh()
            if mlh and mlh["graph_id"] != "no_observations_yet":
                graph_ids, evidences = first_learning_module.evidence_for_each_graph()
                self.add_text(
                    mlh,
                    pos=view_finder_rgba.shape[0],
                    possible_matches=first_learning_module.get_possible_matches(),
                    graph_ids=graph_ids,
                    evidences=evidences,
                )

    @staticmethod
    def _overlay_segmentation(
        rgba: np.ndarray, mask: np.ndarray | None
    ) -> np.ndarray:
        """Tint active segmentation pixels green on a view-finder frame.

        Matches the blend used by ``analysis/visualize_3d.SMTelemetry.overlay``.

        Args:
            rgba: View-finder RGBA image, float in ``[0, 1]`` or uint8.
            mask: ``(H, W)`` segmentation mask, or None.

        Returns:
            A copy of ``rgba`` with the mask tinted, or ``rgba`` unchanged when
            there is nothing to overlay.
        """
        if mask is None or not np.any(mask > 0):
            return rgba

        image = np.asarray(rgba).copy()
        active = np.asarray(mask) > 0
        if image.shape[:2] != active.shape:
            return rgba

        is_uint8 = image.dtype == np.uint8 or (
            np.issubdtype(image.dtype, np.integer) and image.max() > 1
        )
        if is_uint8:
            image = image.astype(np.float32)
            tint_scale = 255.0
            out_dtype = np.uint8
        else:
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image = image / 255.0
            tint_scale = 1.0
            out_dtype = np.float32

        tint = np.zeros_like(image)
        tint[..., 1] = tint_scale
        if image.shape[-1] == 4:
            tint[..., 3] = 0.5 * tint_scale
        image[active] = image[active] * 0.6 + tint[active] * 0.4
        if out_dtype == np.uint8:
            return np.clip(image, 0, 255).astype(np.uint8)
        return np.clip(image, 0.0, 1.0)

    def show_patch(self, first_sensor_depth, filtered_patch_ids=frozenset()):
        """Update the single-patch depth panel.

        Filtered percepts are shown greyed-out with reduced alpha.

        Args:
            first_sensor_depth: Depth image for the primary patch sensor.
            filtered_patch_ids: Patch sensor IDs filtered by attention this step.
        """
        if self.depth_image:
            self.depth_image.remove()
        filtered = bool(
            filtered_patch_ids & {SensorID("patch"), SensorID("patch_0")}
        )
        display, imshow_kwargs = self._filtered_depth_display(
            first_sensor_depth, filtered
        )
        self.depth_image = self.ax[1].imshow(display, **imshow_kwargs)

    def show_patches(self, patch_rgbs, filtered_patch_ids=frozenset()):
        """Update the 3x3 middle panel with RGB images from patch_0..patch_8.

        Filtered percepts are shown greyed-out with reduced alpha.

        Args:
            patch_rgbs: RGB arrays ordered to match ``_NINE_PATCH_IDS``.
            filtered_patch_ids: Patch sensor IDs filtered by attention this step.
        """
        for i, (ax, rgb, patch_id) in enumerate(
            zip(self.patch_axes, patch_rgbs, _NINE_PATCH_IDS)
        ):
            if self.patch_images[i] is not None:
                self.patch_images[i].remove()
            self.patch_images[i] = ax.imshow(
                self._filtered_rgb_display(rgb, patch_id in filtered_patch_ids)
            )

    @staticmethod
    def _filtered_rgb_display(rgb: np.ndarray, filtered: bool) -> np.ndarray:
        """Return an RGB(A) image, greyed and faded when filtered.

        Args:
            rgb: HxWx3 image, values in ``[0, 1]`` or ``[0, 255]``.
            filtered: Whether attention filtered this patch's percept.

        Returns:
            RGB array when not filtered, RGBA when filtered.
        """
        rgb = np.asarray(rgb, dtype=float)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        if not filtered:
            return rgb
        grey = np.mean(rgb, axis=-1, keepdims=True)
        dimmed = (1.0 - _FILTERED_GREY_BLEND) * rgb + _FILTERED_GREY_BLEND * grey
        alpha = np.full((*dimmed.shape[:2], 1), _FILTERED_ALPHA, dtype=float)
        return np.concatenate([dimmed, alpha], axis=-1)

    @staticmethod
    def _filtered_depth_display(
        depth: np.ndarray, filtered: bool
    ) -> tuple[np.ndarray, dict]:
        """Return depth display data and ``imshow`` kwargs.

        Args:
            depth: 2D depth image.
            filtered: Whether attention filtered this patch's percept.

        Returns:
            ``(image, kwargs)`` for ``ax.imshow``. When filtered, ``image`` is
            an RGBA array already greyed and faded; otherwise it is the raw
            depth with a ``viridis_r`` colormap.
        """
        if not filtered:
            return np.asarray(depth), {"cmap": "viridis_r"}

        depth = np.asarray(depth, dtype=float)
        dmin = float(np.nanmin(depth))
        dmax = float(np.nanmax(depth))
        if dmax > dmin:
            normed = (depth - dmin) / (dmax - dmin)
        else:
            normed = np.zeros_like(depth, dtype=float)
        rgba = plt.get_cmap("viridis_r")(normed)
        rgb = rgba[..., :3]
        grey = np.mean(rgb, axis=-1, keepdims=True)
        dimmed = (1.0 - _FILTERED_GREY_BLEND) * rgb + _FILTERED_GREY_BLEND * grey
        alpha = np.full((*dimmed.shape[:2], 1), _FILTERED_ALPHA, dtype=float)
        return np.concatenate([dimmed, alpha], axis=-1), {}

    def show_mlh(self, mlh, mlh_model):
        ax = self.ax[2]
        if not mlh_model:
            ax.set_title("No MLH")
            self._hide_3d_axes(ax)
            return

        ax.cla()
        ax.scatter(
            mlh_model.pos[:, 1],
            mlh_model.pos[:, 0],
            mlh_model.pos[:, 2],
            c="black",
            s=2,
        )
        # add mlh location to the graph
        ax.scatter(
            mlh["location"][1], mlh["location"][0], mlh["location"][2], c="red", s=15
        )
        ax.set_title("MLH")
        ax.set_aspect("equal")
        self._hide_3d_axes(ax)

    def show_attention(
        self,
        attention_system,
        status: str = "",
        inhibited_locations: np.ndarray | None = None,
    ) -> None:
        """Plot the attention system's current voxel filter.

        Shows voxels held in ``attention_system.grid`` after the latest
        ``attention_system.step``, plus inhibit points applied this step
        (lagged one step from LM ``propose_region``). Draws a side view in
        the rightmost top panel and a top view in the panel below it.

        Args:
            attention_system: The model's attention system, or None.
            status: Recognition / inhibit status string for the title.
            inhibited_locations: ``(N, 3)`` inhibit points applied this step.
        """
        if inhibited_locations is None:
            inhibited_locations = np.empty((0, 3))
        else:
            inhibited_locations = np.asarray(inhibited_locations, dtype=float).reshape(
                -1, 3
            )

        centres = np.empty((0, 3))
        weights = np.empty(0)
        voxel_size = (
            attention_system.voxel_size if attention_system is not None else 0.01
        )
        voxel_lifetime = (
            float(attention_system.voxel_lifetime)
            if attention_system is not None
            else 1.0
        )
        if attention_system is not None and len(attention_system.grid) > 0:
            grid = attention_system.grid
            indices = grid.index.to_frame(index=False).to_numpy(dtype=float)
            centres = (indices + _VOXEL_CENTRE_OFFSET) * voxel_size
            weights = grid["weight"].to_numpy(dtype=float)

        inhibiting = len(inhibited_locations) > 0 or "inhibit applied" in status
        title_color = "crimson" if inhibiting else "black"
        title_parts = []
        if len(centres):
            title_parts.append(f"{len(centres)} grid voxels")
        if len(inhibited_locations):
            title_parts.append(f"{len(inhibited_locations)} inhibit applied")
        count_label = ", ".join(title_parts) if title_parts else "empty"

        bound_points = (
            np.vstack([centres, inhibited_locations])
            if len(centres) and len(inhibited_locations)
            else centres
            if len(centres)
            else inhibited_locations
        )

        # Side: look onto XY (elev=90). Top: look onto XZ (elev=0, azim=-90).
        views = (
            (self.ax[3], 90, -90, 2, f"Attention side ({count_label})"),
            (self.ax[4], 0, -90, 1, f"Attention top ({count_label})"),
        )
        for ax, elev, azim, depth_axis, title in views:
            ax.cla()
            ax.view_init(elev=elev, azim=azim)
            if len(centres) == 0 and len(inhibited_locations) == 0:
                ax.set_title(title, color=title_color, fontsize=9)
                ax.text2D(0.5, 0.5, "No voxels", transform=ax.transAxes, ha="center")
                self._hide_3d_axes(ax)
                continue

            # Bias along the camera-forward axis so mplot3d depth-sorts layers apart.
            depth_bias = voxel_size
            if len(inhibited_locations):
                inhibited_plot = inhibited_locations.copy()
                inhibited_plot[:, depth_axis] -= depth_bias
                ax.scatter(
                    inhibited_plot[:, 0],
                    inhibited_plot[:, 1],
                    inhibited_plot[:, 2],
                    c="lightcoral",
                    s=3,
                    alpha=0.25,
                    depthshade=False,
                )

            if len(centres):
                centres_plot = centres.copy()
                centres_plot[:, depth_axis] += depth_bias
                ax.scatter(
                    centres_plot[:, 0],
                    centres_plot[:, 1],
                    centres_plot[:, 2],
                    c=weights,
                    cmap="viridis",
                    s=28,
                    alpha=1.0,
                    vmin=0.0,
                    vmax=voxel_lifetime,
                    depthshade=False,
                    edgecolors="k",
                    linewidths=0.35,
                )

            ax.set_title(title, color=title_color, fontsize=9)
            self._set_attention_limits(ax, bound_points, voxel_size)
            self._hide_3d_axes(ax)

    def _set_attention_limits(self, ax, points: np.ndarray, voxel_size: float) -> None:
        """Keep a padded, monotonically growing view of the attention cloud."""
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        if self._attention_bounds is None:
            self._attention_bounds = (mins.copy(), maxs.copy())
        else:
            self._attention_bounds = (
                np.minimum(self._attention_bounds[0], mins),
                np.maximum(self._attention_bounds[1], maxs),
            )
        bmin, bmax = self._attention_bounds
        span = np.maximum(bmax - bmin, voxel_size)
        # Square the view and add padding so points are not clipped at the edge.
        side = float(np.max(span)) * 1.25
        mid = 0.5 * (bmin + bmax)
        half = 0.5 * max(side, 4.0 * voxel_size)
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_box_aspect([1, 1, 1])

    @staticmethod
    def _hide_3d_axes(ax) -> None:
        """Fully hide a 3D axis frame, panes, ticks, and grid.

        ``Axes3D.set_axis_off()`` alone is unreliable after ``cla()`` / aspect
        changes, so clear each axis component explicitly.
        """
        ax.set_axis_off()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("none")
            axis.pane.set_alpha(0)
            axis.line.set_color((0.0, 0.0, 0.0, 0.0))
            axis.set_ticklabels([])
        # Private flag still used by some matplotlib 3D draws after set_axis_off.
        ax._axis3don = False

    def add_text(
        self,
        mlh,
        pos,
        possible_matches,
        graph_ids,
        evidences,
    ):
        if self.text:
            self.text.remove()
        new_text = r"MLH of first LM: "
        mlh_id = mlh["graph_id"].split("_")
        for word in mlh_id:
            new_text += r"$\bf{" + word + "}$ "
        new_text += f"with evidence {np.round(mlh['evidence'], 2)}\n\n"

        # Highlight 2nd MLH if present
        if len(evidences) > 1:
            top_indices = np.flip(np.argsort(evidences))[0:2]
            second_id = graph_ids[top_indices[1]].split("_")
            new_text += "2nd MLH: "
            for word in second_id:
                new_text += r"$\bf{" + word + "}$ "
            new_text += f"with evidence {np.round(evidences[top_indices[1]], 2)}\n\n"

        new_text += r"$\bf{Possible}$ $\bf{matches:}$"
        for gid, ev in zip(graph_ids, evidences):
            if gid in possible_matches:
                new_text += f"\n{gid}: {np.round(ev, 1)}"

        self.text = self.ax[0].text(0, pos + 30, new_text, va="top")

    def setup_camera_ax(self):
        self.ax[0].set_title("Camera image")
        self.ax[0].set_axis_off()
        self.camera_image = None
        self.text = None

    def setup_sensor_ax(self):
        if self._nine_patch_mode:
            self.patch_images = [None] * len(self.patch_axes)
            for ax, patch_id in zip(self.patch_axes, _NINE_PATCH_IDS):
                ax.set_title(str(patch_id), fontsize=8)
                ax.set_axis_off()
            self.depth_image = None
        else:
            self.ax[1].set_title("Sensor depth image")
            self.ax[1].set_axis_off()
            self.depth_image = None
            self.patch_images = []

    def setup_mlh_ax(self):
        self.ax[2].set_title("MLH")
        self.ax[2].set_aspect("equal")
        self._hide_3d_axes(self.ax[2])

    def setup_attention_ax(self):
        self.ax[3].set_title("Attention side")
        self.ax[3].view_init(elev=90, azim=-90)
        self._hide_3d_axes(self.ax[3])
        self.ax[4].set_title("Attention top")
        self.ax[4].view_init(elev=0, azim=-90)
        self._hide_3d_axes(self.ax[4])
