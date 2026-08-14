# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import pandas as pd

from tbp.monty.cmp import Goal
from tbp.monty.memento import Memento

__all__ = ["AttentionSystemTelemetry"]


class AttentionSystemTelemetry:
    def __init__(self) -> None:
        self.voxel_grids: list[pd.DataFrame] = []
        self.pre_filter_goals: list[list[Goal]] = []
        self.post_filter_goals: list[list[Goal]] = []
        self.filtered_percepts: list[list[str]] = []
        self.inhibitory_senders: list[list[str]] = []

    def reset(self) -> None:
        self.voxel_grids = []
        self.pre_filter_goals = []
        self.post_filter_goals = []
        self.filtered_percepts = []
        self.inhibitory_senders = []

    def percept_filtering(self, filtered: list[str]) -> None:
        """Record which percepts were disabled on one call.

        Args:
            filtered: Sender ids of the percepts whose use_state was disabled.
        """
        self.filtered_percepts.append(list(filtered))

    def region_inhibition(self, senders: list[str]) -> None:
        """Record which modules proposed inhibitory regions in one update.

        Args:
            senders: Sender ids that proposed negative-weight regions.
        """
        self.inhibitory_senders.append(list(senders))

    def voxel_grid(self, grid: pd.DataFrame) -> None:
        self.voxel_grids.append(grid.copy())

    def goal_filtering(self, pre: list[Goal], post: list[Goal]) -> None:
        """Record one step's goals as they entered and left the filter.

        Args:
            pre: The goals handed to the attention system this step.
            post: The goals that survived the voxel grid filter.
        """
        self.pre_filter_goals.append(list(pre))
        self.post_filter_goals.append(list(post))

    def state_dict(self) -> Memento:
        return dict(
            voxel_grids=[
                dict(
                    voxels=grid.index.to_frame(index=False).to_numpy(),
                    weight=grid["weight"].to_numpy(),
                )
                for grid in self.voxel_grids
            ],
            pre_filter_goals=self.pre_filter_goals,
            post_filter_goals=self.post_filter_goals,
            filtered_percepts=self.filtered_percepts,
            inhibitory_senders=self.inhibitory_senders,
        )
