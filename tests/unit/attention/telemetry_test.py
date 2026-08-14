# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import unittest

import numpy as np

from tbp.monty.attention.attention_system import AttentionSystem
from tbp.monty.attention.telemetry import AttentionSystemTelemetry
from tbp.monty.frameworks.models.buffer import BufferEncoder

from .attention_system_test import goal_at, percept_at, region

NEAR_POINT = [0.0, 0, 0]
FAR_POINT = [0.5, 0, 0]


class AttentionSystemTelemetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.telemetry = AttentionSystemTelemetry()
        self.system = AttentionSystem(
            voxel_size=0.01, voxel_lifetime=2, telemetry=self.telemetry
        )

    def step(self, regions: list) -> None:
        """Update the grid and filter goals, as one Monty step does."""
        self.system.update_regions(regions)
        self.system.filter_goals([])

    def test_each_goal_filtering_records_a_grid_snapshot(self) -> None:
        self.step([region(NEAR_POINT, weight=2.0)])
        self.step([region(FAR_POINT, weight=2.0)])
        self.assertEqual(len(self.telemetry.voxel_grids), 2)

    def test_a_snapshot_is_unaffected_by_later_steps(self) -> None:
        self.step([region(NEAR_POINT, weight=2.0)])
        self.step([region(FAR_POINT, weight=2.0)])
        self.assertEqual(len(self.telemetry.voxel_grids[0]), 1)
        self.assertEqual(len(self.telemetry.voxel_grids[1]), 2)

    def test_reset_discards_the_snapshots(self) -> None:
        self.step([region(NEAR_POINT, weight=2.0)])
        self.system.reset()
        self.assertEqual(len(self.telemetry.voxel_grids), 0)

    def test_state_dict_flattens_each_snapshot_into_arrays(self) -> None:
        self.step([region(NEAR_POINT, weight=2.0)])
        self.step([region(NEAR_POINT, FAR_POINT, weight=2.0)])
        snapshot = self.system.state_dict()["voxel_grids"][1]
        np.testing.assert_array_equal(snapshot["voxels"], [[0, 0, 0], [50, 0, 0]])
        np.testing.assert_array_equal(snapshot["weight"], [2, 2])

    def test_an_empty_grid_snapshot_is_exported_empty(self) -> None:
        self.step([region(NEAR_POINT, weight=2.0)])
        # Two empty steps decay the voxel past its lifetime of 2.
        self.step([])
        self.step([])
        snapshot = self.system.state_dict()["voxel_grids"][2]
        self.assertEqual(len(snapshot["voxels"]), 0)
        self.assertEqual(len(snapshot["weight"]), 0)

    def test_state_dict_is_json_encodable(self) -> None:
        self.step([region(NEAR_POINT, weight=2.0)])
        self.step([])
        encoded = json.loads(json.dumps(self.system.state_dict(), cls=BufferEncoder))
        self.assertEqual(encoded["voxel_grids"][0]["voxels"], [[0, 0, 0]])

    def test_a_default_telemetry_is_created_when_none_is_supplied(self) -> None:
        system = AttentionSystem(voxel_size=0.01)
        system.update_regions([region(NEAR_POINT, weight=2.0)])
        system.filter_goals([])
        self.assertEqual(len(system.state_dict()["voxel_grids"]), 1)

    def test_state_dict_carries_the_grid_geometry(self) -> None:
        state = self.system.state_dict()
        self.assertEqual(state["voxel_size"], 0.01)
        self.assertEqual(state["voxel_lifetime"], 2)

    def test_goal_filtering_records_pre_and_post_goals(self) -> None:
        self.system.update_regions([region(NEAR_POINT, weight=2.0)])
        inside = goal_at(NEAR_POINT)
        outside = goal_at([9.0, 9, 9])
        self.system.filter_goals([inside, outside])
        self.assertEqual(self.telemetry.pre_filter_goals, [[inside, outside]])
        self.assertEqual(self.telemetry.post_filter_goals, [[inside]])

    def test_percept_filtering_records_an_entry_per_call(self) -> None:
        # One entry whether or not the grid is populated.
        self.system.filter_percepts([percept_at(NEAR_POINT)])
        self.system.update_regions([region(NEAR_POINT, weight=2.0)])
        self.system.filter_percepts([percept_at(NEAR_POINT)])
        self.assertEqual(self.telemetry.filtered_percepts, [[], []])

    def test_percept_filtering_records_disabled_senders(self) -> None:
        self.system.update_regions([region(NEAR_POINT, weight=-2.0)])
        self.system.filter_percepts([percept_at(NEAR_POINT, sender_id="patch_4")])
        self.assertEqual(self.telemetry.filtered_percepts, [["patch_4"]])

    def test_region_updates_record_inhibitory_senders(self) -> None:
        self.system.update_regions(
            [
                region(NEAR_POINT, weight=2.0, sender_id="patch_0"),
                region(FAR_POINT, weight=-2.0, sender_id="learning_module_3"),
            ]
        )
        self.assertEqual(self.telemetry.inhibitory_senders, [["learning_module_3"]])

    def test_region_updates_record_an_empty_entry_when_none_inhibit(self) -> None:
        self.system.update_regions([region(NEAR_POINT, weight=2.0)])
        self.assertEqual(self.telemetry.inhibitory_senders, [[]])

    def test_state_dict_carries_the_new_telemetry(self) -> None:
        self.system.update_regions([region(NEAR_POINT, weight=-2.0)])
        self.system.filter_percepts([percept_at(NEAR_POINT, sender_id="patch_4")])
        state = self.system.state_dict()
        self.assertEqual(state["inhibitory_senders"], [["SM_0"]])
        self.assertEqual(state["filtered_percepts"], [["patch_4"]])
