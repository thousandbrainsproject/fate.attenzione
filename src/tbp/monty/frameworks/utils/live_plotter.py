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
_NINE_PATCH_IDS = tuple(
    SensorID(f"patch_{i}") for i in (1, 2, 3, 4, 0, 5, 6, 7, 8)
)


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
    """

    def __init__(self):
        self._nine_patch_mode = False
        self.save_dir: Path | None = None

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
        if self._nine_patch_mode:
            self.fig = plt.figure(figsize=(10, 5.5))
            self.fig.subplots_adjust(top=0.9, wspace=0.2, hspace=0.15)
            # Narrow middle column, with vertical padding so the 3x3 stays compact.
            gs = gridspec.GridSpec(
                3,
                3,
                figure=self.fig,
                width_ratios=[1.2, 0.65, 1.2],
                height_ratios=[0.18, 0.64, 0.18],
            )
            camera_ax = self.fig.add_subplot(gs[:, 0])
            patch_gs = gs[1, 1].subgridspec(3, 3, wspace=0.05, hspace=0.2)
            self.patch_axes = [
                self.fig.add_subplot(patch_gs[row, col])
                for row in range(3)
                for col in range(3)
            ]
            mlh_ax = self.fig.add_subplot(gs[:, 2], projection="3d")
            # Keep self.ax[0]/[2] as camera / MLH; middle slot unused in 9-patch mode.
            self.ax = [camera_ax, None, mlh_ax]
        else:
            self.fig = plt.figure(figsize=(9, 6))
            self.fig.subplots_adjust(top=1.1)
            self.ax = [
                self.fig.add_subplot(1, 3, 1),
                self.fig.add_subplot(1, 3, 2),
                self.fig.add_subplot(1, 3, 3, projection="3d"),
            ]
            self.patch_axes = []

        self.setup_camera_ax()
        self.setup_sensor_ax()
        self.setup_mlh_ax()

        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

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
            and optionally a list of RGB images for patch_0..patch_8.
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
        return (
            first_learning_module,
            first_sensor_module_raw_observations,
            first_sensor_depth,
            view_finder_rgba,
            mlh,
            mlh_model,
            patch_rgbs,
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
        step: int,
        is_saccade_on_image_data_loader=False,
    ) -> None:
        self.fig.suptitle(f"Observation at step {step}")
        self.show_view_finder(
            first_sensor_module_raw_observations,
            first_learning_module,
            first_sensor_depth,
            view_finder_rgba,
            is_saccade_on_image_data_loader,
        )
        if patch_rgbs is not None:
            self.show_patches(patch_rgbs)
        else:
            self.show_patch(first_sensor_depth)
        if mlh_model:
            self.show_mlh(mlh, mlh_model)
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

    def show_patches(self, patch_rgbs):
        """Update the 3x3 middle panel with RGB images from patch_0..patch_8."""
        for i, (ax, rgb) in enumerate(zip(self.patch_axes, patch_rgbs)):
            if self.patch_images[i] is not None:
                self.patch_images[i].remove()
            self.patch_images[i] = ax.imshow(rgb)

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
