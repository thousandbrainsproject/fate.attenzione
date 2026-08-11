# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

import numpy as np

from tbp.monty.attention.attention_system import AttentionSystem
from tbp.monty.cmp import AttentionWeight, Goal

# Two points inside one voxel, and a third far enough away to occupy its own.
NEAR_POINTS = ([0.0, 0, 0], [0.005, 0, 0])
FAR_POINT = [0.5, 0, 0]
# The voxels those points fall in, at voxel_size=0.01.
NEAR_VOXEL = (0, 0, 0)
FAR_VOXEL = (50, 0, 0)


def goal_at(location, confidence: float = 0.5) -> Goal:
    """Build a goal at the given location.

    Returns:
        A goal whose only meaningful properties here are its location and
        confidence.

    """
    return Goal(
        location=None if location is None else np.asarray(location, dtype=float),
        morphological_features=None,
        non_morphological_features=None,
        confidence=confidence,
        use_state=False,
        sender_id="SM_0",
        sender_type="SM",
        goal_tolerances=None,
    )


def attention_weight_at(location, weight: float = 1.0) -> AttentionWeight:
    """Build an attention weight at the given location.

    Returns:
        An attention weight whose only meaningful properties here are its
        location and weight.

    """
    return AttentionWeight(
        location=None if location is None else np.asarray(location, dtype=float),
        weight=weight,
        sender_id="SM_0",
        sender_type="SM",
    )


def region(*locations, weight: float = 1.0) -> list[AttentionWeight]:
    """Build a region from the given locations.

    Returns:
        One region: a list with one attention weight per location.

    """
    return [attention_weight_at(location, weight) for location in locations]


def column_by_voxel(
    system: AttentionSystem, column: str
) -> dict[tuple[int, int, int], float]:
    """Map each occupied voxel to one of its column values.

    Returns:
        Voxel coordinate to value, for every voxel the system holds.

    """
    data = system.grid
    voxels = [tuple(int(c) for c in index) for index in data.index]
    return dict(zip(voxels, data[column].to_numpy().ravel().tolist()))


def weights_by_voxel(system: AttentionSystem) -> dict[tuple[int, int, int], float]:
    """Map each occupied voxel to its current weight.

    Returns:
        Voxel coordinate to weight, for every voxel the system holds.

    """
    return column_by_voxel(system, "weight")


def counts_by_voxel(system: AttentionSystem) -> dict[tuple[int, int, int], int]:
    """Map each occupied voxel to how many steps it has been proposed.

    Returns:
        Voxel coordinate to count, for every voxel the system holds.

    """
    return column_by_voxel(system, "count")


class AttentionSystemGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01)

    def test_locations_sharing_a_voxel_collapse_to_one_row(self) -> None:
        self.system.step([], [region(*NEAR_POINTS, FAR_POINT)])
        self.assertEqual(len(self.system.grid), 2)

    def test_no_regions_yield_an_empty_grid(self) -> None:
        self.system.step([], [])
        self.assertEqual(len(self.system.grid), 0)

    def test_empty_regions_yield_an_empty_grid(self) -> None:
        self.system.step([], [[], []])
        self.assertEqual(len(self.system.grid), 0)

    def test_attention_weights_without_a_location_are_not_voxelized(self) -> None:
        self.system.step([], [[attention_weight_at(None)]])
        self.assertEqual(len(self.system.grid), 0)

    def test_a_step_adds_to_the_grid_rather_than_replacing_it(self) -> None:
        # Weight 2 so the near voxel survives the decay tick of the second step.
        self.system.step([], [region(*NEAR_POINTS, weight=2.0)])
        self.system.step([], [region(FAR_POINT)])
        self.assertEqual(
            sorted(weights_by_voxel(self.system)), [NEAR_VOXEL, FAR_VOXEL]
        )

    def test_regions_from_different_modules_merge_into_one_grid(self) -> None:
        self.system.step([], [region(*NEAR_POINTS), region(FAR_POINT)])
        self.assertEqual(
            sorted(weights_by_voxel(self.system)), [NEAR_VOXEL, FAR_VOXEL]
        )

    def test_reset_discards_the_grid(self) -> None:
        self.system.step([], [region(*NEAR_POINTS)])
        self.system.reset()
        self.assertEqual(len(self.system.grid), 0)


class AttentionSystemWeightTest(unittest.TestCase):
    """Signed voxel weights decay toward zero and expire when they reach it."""

    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01, voxel_lifetime=3)

    def observe_near(self, weight: float = 3.0) -> None:
        """Observe the near voxel."""
        self.system.step([], [region(NEAR_POINTS[0], weight=weight)])

    def observe_far(self) -> None:
        """Observe the far voxel."""
        self.system.step([], [region(FAR_POINT, weight=3.0)])

    def test_a_fresh_voxel_carries_the_proposed_weight(self) -> None:
        self.observe_near(weight=2.0)
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 2.0)

    def test_a_voxels_weight_is_the_mean_of_its_proposals(self) -> None:
        self.system.step(
            [],
            [
                [
                    attention_weight_at(NEAR_POINTS[0], 1.0),
                    attention_weight_at(NEAR_POINTS[1], 2.0),
                ]
            ],
        )
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 1.5)

    def test_weight_is_capped_at_the_voxel_lifetime(self) -> None:
        self.observe_near(weight=100.0)
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 3.0)

    def test_voxel_lifetime_is_exposed(self) -> None:
        self.assertEqual(AttentionSystem(voxel_lifetime=9).voxel_lifetime, 9)

    def test_voxel_lifetime_must_be_positive(self) -> None:
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                AttentionSystem(voxel_lifetime=bad)

    def test_an_unobserved_voxel_is_remembered(self) -> None:
        self.observe_near()
        self.observe_far()
        self.assertEqual(set(weights_by_voxel(self.system)), {NEAR_VOXEL, FAR_VOXEL})

    def test_an_unobserved_voxel_decays_by_one_step(self) -> None:
        self.observe_near()
        self.observe_far()
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 2.0)

    def test_a_re_observed_voxel_accumulates_up_to_the_cap(self) -> None:
        self.observe_near()
        self.observe_far()
        self.observe_near()
        # decayed 3 -> 1 over two steps, + fresh 3, capped at lifetime 3.
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 3.0)

    def test_a_voxel_expires_once_its_weight_decays_to_zero(self) -> None:
        self.observe_near()
        for _ in range(3):
            self.observe_far()
        self.assertEqual(set(weights_by_voxel(self.system)), {FAR_VOXEL})

    def test_observing_nothing_still_decays_the_grid(self) -> None:
        self.observe_near()
        self.system.step([], [])
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 2.0)

    def test_the_grid_empties_once_everything_expires(self) -> None:
        self.observe_near()
        for _ in range(3):
            self.system.step([], [])
        self.assertEqual(len(self.system.grid), 0)


class AttentionSystemNegativeWeightTest(unittest.TestCase):
    """Negative weights persist, decay toward zero, and win contested voxels."""

    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01, voxel_lifetime=3)

    def test_a_negative_voxel_persists_and_decays_toward_zero(self) -> None:
        self.system.step([], [region(NEAR_POINTS[0], weight=-3.0)])
        trace = [weights_by_voxel(self.system).get(NEAR_VOXEL)]
        for _ in range(3):
            self.system.step([], [])
            trace.append(weights_by_voxel(self.system).get(NEAR_VOXEL))
        self.assertEqual(trace, [-3.0, -2.0, -1.0, None])

    def test_negative_wins_a_contested_voxel_regardless_of_point_counts(self) -> None:
        dense_positive = [
            attention_weight_at(NEAR_POINTS[0], 3.0) for _ in range(10)
        ]
        one_negative = [attention_weight_at(NEAR_POINTS[1], -3.0)]
        self.system.step([], [one_negative, dense_positive])
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], -3.0)

    def test_contested_weight_is_the_mean_of_the_negative_proposals(self) -> None:
        self.system.step(
            [],
            [
                [
                    attention_weight_at(NEAR_POINTS[0], -1.0),
                    attention_weight_at(NEAR_POINTS[1], -2.0),
                    attention_weight_at(NEAR_POINTS[0], 3.0),
                ]
            ],
        )
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], -1.5)

    def test_re_proposed_inhibition_accumulates_down_to_the_cap(self) -> None:
        self.system.step([], [region(NEAR_POINTS[0], weight=-3.0)])
        self.system.step([], [region(NEAR_POINTS[0], weight=-3.0)])
        # decayed -3 -> -2, + fresh -3, capped at -lifetime -3.
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], -3.0)

    def test_inhibition_overrides_a_remembered_positive_voxel(self) -> None:
        self.system.step([], [region(NEAR_POINTS[0], weight=3.0)])
        self.system.step([], [region(NEAR_POINTS[0], weight=-3.0)])
        # decayed 3 -> 2, + fresh -3 -> net -1: repulsive.
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], -1.0)


class AttentionSystemCountTest(unittest.TestCase):
    """Count tallies how many steps a voxel has been proposed."""

    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01, voxel_lifetime=3)

    def observe_near(self) -> None:
        """Observe the near voxel."""
        self.system.step([], [region(NEAR_POINTS[0], weight=3.0)])

    def observe_far(self) -> None:
        """Observe the far voxel."""
        self.system.step([], [region(FAR_POINT, weight=3.0)])

    def test_a_newly_seen_voxel_starts_at_one(self) -> None:
        self.observe_near()
        self.assertEqual(counts_by_voxel(self.system)[NEAR_VOXEL], 1)

    def test_re_observing_adds_to_the_count(self) -> None:
        for expected in (1, 2, 3):
            self.observe_near()
            self.assertEqual(counts_by_voxel(self.system)[NEAR_VOXEL], expected)

    def test_one_steps_points_count_as_a_single_sighting(self) -> None:
        # Many points, even from several modules, are one step's sighting.
        self.system.step(
            [], [region(NEAR_POINTS[0]), region(NEAR_POINTS[1])]
        )
        self.assertEqual(counts_by_voxel(self.system)[NEAR_VOXEL], 1)

    def test_an_unobserved_voxel_keeps_its_count(self) -> None:
        self.observe_near()
        self.observe_near()
        self.observe_far()
        counts = counts_by_voxel(self.system)
        self.assertEqual(counts[NEAR_VOXEL], 2)
        self.assertEqual(counts[FAR_VOXEL], 1)

    def test_count_restarts_after_a_voxel_expires(self) -> None:
        # An expired voxel is forgotten, so its tally starts over.
        self.observe_near()
        self.observe_near()
        for _ in range(3):
            self.system.step([], [])
        self.assertEqual(len(self.system.grid), 0)
        self.observe_near()
        self.assertEqual(counts_by_voxel(self.system)[NEAR_VOXEL], 1)

    def test_count_stays_an_integer_across_steps(self) -> None:
        self.observe_near()
        self.observe_near()
        self.assertEqual(self.system.grid["count"].to_numpy().dtype, np.int32)


class AttentionSystemContainsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01)
        self.system.step([], [region(*NEAR_POINTS, FAR_POINT)])

    def test_many_locations_yield_an_array(self) -> None:
        result = self.system.contains_points(np.array([[0.0, 0, 0], [9.0, 9, 9]]))
        np.testing.assert_array_equal(result, [True, False])

    def test_a_single_flat_point_is_accepted(self) -> None:
        # Normalized to (1, 3), so the result is still an array.
        np.testing.assert_array_equal(
            self.system.contains_points(np.array([0.0, 0, 0])), [True]
        )

    def test_any_location_in_an_occupied_voxel_is_contained(self) -> None:
        # A different point in the same voxel as an observed one.
        np.testing.assert_array_equal(
            self.system.contains_points(np.array([[0.009, 0, 0]])), [True]
        )

    def test_an_empty_grid_contains_nothing(self) -> None:
        empty = AttentionSystem(voxel_size=0.01)
        np.testing.assert_array_equal(
            empty.contains_points(np.array([[0.0, 0, 0]])), [False]
        )

    def test_weights_at_points_reads_held_voxels_and_defaults_to_zero(self) -> None:
        weights = self.system.weights_at_points(
            np.array([[0.0, 0, 0], [9.0, 9, 9]])
        )
        np.testing.assert_array_equal(weights, [1.0, 0.0])


class AttentionSystemWeighTest(unittest.TestCase):
    """Step re-weights goal confidences by the voxels they fall in."""

    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01, voxel_lifetime=4)

    def test_full_repulsion_zeroes_a_goals_confidence(self) -> None:
        goal = goal_at(NEAR_POINTS[0], confidence=0.8)
        (returned,) = self.system.step(
            [goal], [region(NEAR_POINTS[1], weight=-4.0)]
        )
        self.assertEqual(returned.confidence, 0.0)

    def test_partial_repulsion_scales_confidence_down(self) -> None:
        goal = goal_at(NEAR_POINTS[0], confidence=0.8)
        (returned,) = self.system.step(
            [goal], [region(NEAR_POINTS[1], weight=-2.0)]
        )
        # scale = 1 + (-2 / 4) = 0.5
        self.assertAlmostEqual(returned.confidence, 0.4)

    def test_attraction_scales_confidence_headroom_up(self) -> None:
        goal = goal_at(NEAR_POINTS[0], confidence=0.4)
        (returned,) = self.system.step(
            [goal], [region(NEAR_POINTS[1], weight=4.0)]
        )
        # conf + (4/4) * conf * (1 - conf) = 0.4 + 0.4 * 0.6
        self.assertAlmostEqual(returned.confidence, 0.64)

    def test_boosting_never_saturates_the_ranking(self) -> None:
        # A full-attraction boost must preserve the goals' salience ordering
        # rather than clipping them all to 1.0.
        goals = [
            goal_at(NEAR_POINTS[0], confidence=0.6),
            goal_at(NEAR_POINTS[0], confidence=0.9),
        ]
        low, high = self.system.step(
            goals, [region(NEAR_POINTS[1], weight=4.0)]
        )
        self.assertLess(low.confidence, high.confidence)
        self.assertLess(high.confidence, 1.0)

    def test_a_goal_outside_any_voxel_is_unchanged(self) -> None:
        goal = goal_at([9.0, 9, 9], confidence=0.4)
        (returned,) = self.system.step(
            [goal], [region(NEAR_POINTS[0], weight=4.0)]
        )
        self.assertEqual(returned.confidence, 0.4)

    def test_goals_pass_through_while_the_grid_is_empty(self) -> None:
        goals = [goal_at(NEAR_POINTS[0]), goal_at([9.0, 9, 9])]
        self.assertEqual(self.system.step(goals, []), goals)

    def test_goals_without_a_location_pass_through(self) -> None:
        unlocated = goal_at(None)
        returned = self.system.step(
            [unlocated], [region(NEAR_POINTS[0], weight=-4.0)]
        )
        self.assertEqual(returned, [unlocated])

    def test_all_goals_are_returned_not_filtered(self) -> None:
        goals = [
            goal_at(NEAR_POINTS[0], confidence=0.8),
            goal_at([9.0, 9, 9], confidence=0.8),
        ]
        returned = self.system.step(
            goals, [region(NEAR_POINTS[1], weight=-4.0)]
        )
        self.assertEqual(len(returned), 2)

    def test_weighing_copies_goals_rather_than_mutating_them(self) -> None:
        goal = goal_at(NEAR_POINTS[0], confidence=0.8)
        (returned,) = self.system.step(
            [goal], [region(NEAR_POINTS[1], weight=-4.0)]
        )
        self.assertEqual(goal.confidence, 0.8)
        self.assertIsNot(returned, goal)

    def test_goals_are_weighed_against_the_updated_grid(self) -> None:
        # The region arrives on the same step as the goal it boosts.
        goal = goal_at(NEAR_POINTS[1], confidence=0.4)
        (returned,) = self.system.step(
            [goal], [region(NEAR_POINTS[0], weight=4.0)]
        )
        self.assertAlmostEqual(returned.confidence, 0.64)
