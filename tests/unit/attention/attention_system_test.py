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
from tbp.monty.cmp import AttentionWeight, Goal, Message

# Two points inside one voxel, and a third far enough away to occupy its own.
NEAR_POINTS = ([0.0, 0, 0], [0.005, 0, 0])
FAR_POINT = [0.5, 0, 0]
# The voxels those points fall in, at voxel_size=0.01.
NEAR_VOXEL = (0, 0, 0)
FAR_VOXEL = (50, 0, 0)


def goal_at(location) -> Goal:
    """Build a goal at the given location.

    Returns:
        A goal whose only meaningful property here is its location.

    """
    return Goal(
        location=None if location is None else np.asarray(location, dtype=float),
        morphological_features=None,
        non_morphological_features=None,
        confidence=0.5,
        use_state=False,
        sender_id="SM_0",
        sender_type="SM",
        goal_tolerances=None,
    )


def percept_at(location, sender_id: str = "patch_0") -> Message:
    """Build a usable percept at the given location.

    Returns:
        A message with ``use_state=True`` and the minimal morphological
        features that a usable message must carry.

    """
    return Message(
        location=np.asarray(location, dtype=float),
        morphological_features={
            "pose_vectors": np.eye(3),
            "pose_fully_defined": True,
        },
        non_morphological_features={},
        confidence=1.0,
        use_state=True,
        sender_id=sender_id,
        sender_type="SM",
    )


def attention_weight_at(
    location, weight: float = 1.0, sender_id: str = "SM_0"
) -> AttentionWeight:
    """Build an attention weight at the given location.

    Returns:
        An attention weight whose only meaningful properties here are its
        location, weight, and sender.

    """
    return AttentionWeight(
        location=None if location is None else np.asarray(location, dtype=float),
        weight=weight,
        sender_id=sender_id,
        sender_type="SM",
    )


def region(
    *locations, weight: float = 1.0, sender_id: str = "SM_0"
) -> list[AttentionWeight]:
    """Build a region from the given locations.

    Returns:
        One region: a list with one attention weight per location.

    """
    return [
        attention_weight_at(location, weight, sender_id) for location in locations
    ]


def weights_by_voxel(system: AttentionSystem) -> dict[tuple[int, int, int], float]:
    """Map each occupied voxel to its current weight.

    Returns:
        Voxel coordinate to weight, for every voxel the system holds.

    """
    data = system.grid
    voxels = [tuple(int(c) for c in index) for index in data.index]
    return dict(zip(voxels, data["weight"].to_numpy().ravel().tolist()))


class AttentionSystemGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01)

    def test_locations_sharing_a_voxel_collapse_to_one_row(self) -> None:
        self.system.update_regions([region(*NEAR_POINTS, FAR_POINT)])
        self.assertEqual(len(self.system.grid), 2)

    def test_no_regions_yield_an_empty_grid(self) -> None:
        self.system.update_regions([])
        self.assertEqual(len(self.system.grid), 0)

    def test_empty_regions_yield_an_empty_grid(self) -> None:
        self.system.update_regions([[], []])
        self.assertEqual(len(self.system.grid), 0)

    def test_attention_weights_without_a_location_are_not_voxelized(self) -> None:
        self.system.update_regions([[attention_weight_at(None)]])
        self.assertEqual(len(self.system.grid), 0)

    def test_an_update_adds_to_the_grid_rather_than_replacing_it(self) -> None:
        # Weight 2 so the near voxel survives the decay tick of the second
        # update.
        self.system.update_regions([region(*NEAR_POINTS, weight=2.0)])
        self.system.update_regions([region(FAR_POINT)])
        self.assertEqual(sorted(weights_by_voxel(self.system)), [NEAR_VOXEL, FAR_VOXEL])

    def test_regions_from_different_modules_merge_into_one_grid(self) -> None:
        self.system.update_regions([region(*NEAR_POINTS), region(FAR_POINT)])
        self.assertEqual(sorted(weights_by_voxel(self.system)), [NEAR_VOXEL, FAR_VOXEL])

    def test_reset_discards_the_grid(self) -> None:
        self.system.update_regions([region(*NEAR_POINTS)])
        self.system.reset()
        self.assertEqual(len(self.system.grid), 0)


class AttentionSystemWeightTest(unittest.TestCase):
    """Voxel weights start at the proposed mean, decay, and expire at zero."""

    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01, voxel_lifetime=3)

    def observe_near(self, weight: float = 3.0) -> None:
        """Observe the near voxel."""
        self.system.update_regions([region(NEAR_POINTS[0], weight=weight)])

    def observe_far(self) -> None:
        """Observe the far voxel."""
        self.system.update_regions([region(FAR_POINT, weight=3.0)])

    def test_a_fresh_voxel_carries_the_proposed_weight(self) -> None:
        self.observe_near(weight=2.0)
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 2.0)

    def test_a_voxels_weight_is_the_mean_of_its_proposals(self) -> None:
        self.system.update_regions(
            [
                [
                    attention_weight_at(NEAR_POINTS[0], 1.0),
                    attention_weight_at(NEAR_POINTS[1], 2.0),
                ]
            ],
        )
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 1.5)

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
        # decayed 3 -> 1 over two updates, + fresh 3, capped at lifetime 3.
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 3.0)

    def test_a_voxel_expires_once_its_weight_decays_to_zero(self) -> None:
        self.observe_near()
        for _ in range(3):
            self.observe_far()
        self.assertEqual(set(weights_by_voxel(self.system)), {FAR_VOXEL})

    def test_observing_nothing_still_decays_the_grid(self) -> None:
        self.observe_near()
        self.system.update_regions([])
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], 2.0)

    def test_the_grid_empties_once_everything_expires(self) -> None:
        self.observe_near()
        for _ in range(3):
            self.system.update_regions([])
        self.assertEqual(len(self.system.grid), 0)


class AttentionSystemInhibitionTest(unittest.TestCase):
    """Negative weights persist as inhibition and decay toward zero."""

    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01, voxel_lifetime=3)

    def test_a_negative_proposal_creates_a_negative_voxel(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=-3.0)])
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], -3.0)

    def test_a_negative_voxel_decays_toward_zero(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=-3.0)])
        self.system.update_regions([])
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], -2.0)

    def test_a_negative_voxel_expires_once_it_decays_to_zero(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=-3.0)])
        for _ in range(3):
            self.system.update_regions([])
        self.assertEqual(len(self.system.grid), 0)

    def test_inhibition_sums_with_a_remembered_positive_weight(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=3.0)])
        self.system.update_regions([region(NEAR_POINTS[0], weight=-3.0)])
        # remembered 3 decays to 2, + fresh -3 = -1.
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], -1.0)

    def test_a_re_observed_negative_voxel_is_capped_at_the_lifetime(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=-3.0)])
        self.system.update_regions([region(NEAR_POINTS[0], weight=-3.0)])
        # decayed -3 -> -2, + fresh -3 = -5, clipped to -lifetime.
        self.assertEqual(weights_by_voxel(self.system)[NEAR_VOXEL], -3.0)


class AttentionSystemContainsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01)
        self.system.update_regions([region(*NEAR_POINTS, FAR_POINT)])

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


class AttentionSystemFilterGoalsTest(unittest.TestCase):
    """Goals are kept and re-weighted by the voxel they fall in."""

    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01, voxel_lifetime=3)

    def test_goals_outside_the_grid_are_dropped(self) -> None:
        self.system.update_regions([region(*NEAR_POINTS)])
        inside = goal_at(NEAR_POINTS[0])
        outside = goal_at([9.0, 9, 9])
        self.assertEqual(self.system.filter_goals([inside, outside]), [inside])

    def test_goals_are_filtered_against_a_fresh_region(self) -> None:
        # The region arrives on the same step as the goal it admits.
        self.system.update_regions([region(NEAR_POINTS[0])])
        goal = goal_at(NEAR_POINTS[1])
        self.assertEqual(self.system.filter_goals([goal]), [goal])

    def test_goals_pass_through_while_the_grid_is_empty(self) -> None:
        goals = [goal_at(NEAR_POINTS[0]), goal_at([9.0, 9, 9])]
        self.assertEqual(self.system.filter_goals(goals), goals)

    def test_goals_pass_through_while_the_grid_is_all_negative(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=-3.0)])
        goals = [goal_at(NEAR_POINTS[0]), goal_at([9.0, 9, 9])]
        self.assertEqual(self.system.filter_goals(goals), goals)
        self.assertEqual(goals[0].confidence, 0.5)

    def test_goals_without_a_location_pass_through(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0])])
        unlocated = goal_at(None)
        returned = self.system.filter_goals([goal_at([9.0, 9, 9]), unlocated])
        self.assertEqual(returned, [unlocated])

    def test_a_positive_voxel_boosts_a_goals_confidence(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=3.0)])
        goal = goal_at(NEAR_POINTS[0])
        self.system.filter_goals([goal])
        self.assertGreater(goal.confidence, 0.5)

    def test_a_negative_voxel_downweights_a_goals_confidence(self) -> None:
        # A positive voxel elsewhere keeps the grid off the all-negative
        # pass-through path.
        self.system.update_regions(
            [region(NEAR_POINTS[0], weight=-3.0), region(FAR_POINT, weight=3.0)]
        )
        goal = goal_at(NEAR_POINTS[0])
        self.assertEqual(self.system.filter_goals([goal]), [goal])
        self.assertLess(goal.confidence, 0.5)

    def test_a_remembered_voxel_still_admits_goals(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=3.0)])
        self.system.update_regions([region(FAR_POINT, weight=3.0)])
        goal = goal_at(NEAR_POINTS[0])
        # The near voxel was not re-observed, but has not expired either.
        self.assertEqual(self.system.filter_goals([goal]), [goal])

    def test_an_expired_voxel_no_longer_admits_goals(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=3.0)])
        for _ in range(3):
            self.system.update_regions([region(FAR_POINT, weight=3.0)])
        # The near voxel decayed past its lifetime before filtering.
        self.assertEqual(self.system.filter_goals([goal_at(NEAR_POINTS[0])]), [])


class AttentionSystemFilterPerceptsTest(unittest.TestCase):
    """Percepts in inhibited voxels are disabled in place."""

    def setUp(self) -> None:
        self.system = AttentionSystem(voxel_size=0.01, voxel_lifetime=3)

    def test_a_percept_in_a_negative_voxel_is_disabled(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=-3.0)])
        percept = percept_at(NEAR_POINTS[0])
        self.assertEqual(self.system.filter_percepts([percept]), [percept])
        self.assertFalse(percept.use_state)

    def test_a_percept_in_a_positive_voxel_is_untouched(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=3.0)])
        percept = percept_at(NEAR_POINTS[0])
        self.system.filter_percepts([percept])
        self.assertTrue(percept.use_state)

    def test_a_percept_outside_the_grid_is_untouched(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=-3.0)])
        percept = percept_at([9.0, 9, 9])
        self.system.filter_percepts([percept])
        self.assertTrue(percept.use_state)

    def test_percepts_pass_through_while_the_grid_is_empty(self) -> None:
        percept = percept_at(NEAR_POINTS[0])
        self.assertEqual(self.system.filter_percepts([percept]), [percept])
        self.assertTrue(percept.use_state)

    def test_none_entries_are_tolerated(self) -> None:
        self.system.update_regions([region(NEAR_POINTS[0], weight=-3.0)])
        percept = percept_at(NEAR_POINTS[0])
        self.assertEqual(
            self.system.filter_percepts([None, percept]), [None, percept]
        )
        self.assertFalse(percept.use_state)
