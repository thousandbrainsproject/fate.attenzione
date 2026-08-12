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

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Monty, Observations
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.plot_utils import add_patch_outline_to_view_finder

# turn interactive plotting off -- call plt.show() to open all figures
plt.ioff()

# Voxel coordinates are lower corners; offset to centres for plotting.
_VOXEL_CENTRE_OFFSET = 0.5


class LivePlotter:
    """Class for plotting sensor observations during an experiment.

    Set the `show_sensor_output` flag in the experiment config to True to enable live
    plotting.

    WARNING: This plotter makes a number of assumptions right now. For example, it
    assumes that
    - sensor with ID "view_finder" exists
    - sensor with ID "patch" exists
    - "rgba" modality in "view_finder" sensor observation
    - "depth" modality in "patch" sensor observation
    """

    def __init__(self):
        self.save_dir: Path | None = None
        # Inhibit points held on the LM after terminal checks are only applied by
        # the attention system on the *next* Monty step. Buffer them so the plot
        # shows inhibit on the step it actually affected the grid.
        self._inhibit_applied_locations: np.ndarray = np.empty((0, 3))
        # Growing axis limits so the attention panel does not jump/zoom each step.
        self._attention_bounds: tuple[np.ndarray, np.ndarray] | None = None

    def initialize_online_plotting(self, save_dir: Path | str | None = None):
        # Build mixed 2D/3D axes explicitly. plt.subplots() would leave orphaned 2D
        # axes behind if we later replace slots with projection="3d".
        # Keep the original 1x4 row in place; add a second attention view under
        # the rightmost panel only.
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
        self.setup_camera_ax()
        self.setup_sensor_ax()
        self.setup_mlh_ax()
        self.setup_attention_ax()

        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self._inhibit_applied_locations = np.empty((0, 3))
        self._attention_bounds = None

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
            observations, the patch depth, the view finder rgba, the MLH, the
            MLH model graph, and the attention system (or None).
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

        first_sensor_depth = observation[first_sensor_module_agent_id][
            first_sensor_module_id
        ]["depth"]
        view_finder_rgba = observation[first_sensor_module_agent_id][
            SensorID("view_finder")
        ]["rgba"]
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
        return (
            first_learning_module,
            first_sensor_module_raw_observations,
            first_sensor_depth,
            view_finder_rgba,
            mlh,
            mlh_model,
            attention_system,
        )

    def show_observations(
        self,
        first_learning_module,
        first_sensor_module_raw_observations,
        first_sensor_depth,
        view_finder_rgba,
        mlh,
        mlh_model,
        attention_system,
        step: int,
        is_saccade_on_image_data_loader=False,
    ) -> None:
        self.show_view_finder(
            first_sensor_module_raw_observations,
            first_learning_module,
            first_sensor_depth,
            view_finder_rgba,
            is_saccade_on_image_data_loader,
        )
        self.show_patch(first_sensor_depth)
        if mlh_model:
            self.show_mlh(mlh, mlh_model)
        # propose_region after this step is what will be applied next step. What
        # attention already applied this step is the buffered previous proposal.
        pending_inhibit = self._inhibited_locations(first_learning_module)
        applied_inhibit = self._inhibit_applied_locations
        self._inhibit_applied_locations = pending_inhibit
        status = self._recognition_status(
            first_learning_module, inhibit_applied=len(applied_inhibit) > 0
        )
        title = f"Observation at step {step}"
        if status:
            title = f"{title}\n{status}"
        self.fig.suptitle(
            title,
            color="crimson" if "inhibit applied" in status else "black",
        )
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

    def _inhibited_locations(self, first_learning_module) -> np.ndarray:
        """Locations from the LM's current negative attention region.

        Returns:
            An ``(N, 3)`` array of inhibited locations, or empty when none.
        """
        if not hasattr(first_learning_module, "propose_region"):
            return np.empty((0, 3))
        region = first_learning_module.propose_region() or []
        locations = [
            aw.location
            for aw in region
            if aw.location is not None and aw.weight < 0
        ]
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
    ):
        if self.camera_image:
            self.camera_image.remove()

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

    def show_patch(self, first_sensor_depth):
        if self.depth_image:
            self.depth_image.remove()
        self.depth_image = self.ax[1].imshow(
            first_sensor_depth,
            cmap="viridis_r",
        )
        # self.colorbar.update_normal(self.depth_image)

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
        the original rightmost panel and a top view in a new panel below it.

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

        # Side: look onto XY (elev=90). Top: look onto XZ (elev=0, azim=-90) —
        # the latter is the previous single-panel view.
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

    def _set_attention_limits(
        self, ax, points: np.ndarray, voxel_size: float
    ) -> None:
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
        self.ax[1].set_title("Sensor depth image")
        self.ax[1].set_axis_off()
        self.depth_image = None

    def setup_mlh_ax(self):
        self.ax[2].set_title("MLH")
        self._hide_3d_axes(self.ax[2])

    def setup_attention_ax(self):
        self.ax[3].set_title("Attention side")
        self.ax[3].view_init(elev=90, azim=-90)
        self._hide_3d_axes(self.ax[3])
        self.ax[4].set_title("Attention top")
        self.ax[4].view_init(elev=0, azim=-90)
        self._hide_3d_axes(self.ax[4])
