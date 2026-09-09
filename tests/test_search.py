"""Tests for belief state, mass propagation, and search foundations."""

from __future__ import annotations

import math
import unittest
from typing import Any
import numpy as np
import torch

from icgs.configuration.method import MethodConfig
from icgs.state.physical import PhysicalState
from icgs.contracts.method import (
    CommandPrefix,
    EvaluationOutput,
    PhysicalPrediction,
    PlanningResult,
    TerminalProbabilities,
    TimedCommand,
)
from icgs.algorithms.planning.belief import (
    BeliefNode,
    Hypothesis,
    leaf_return,
    propagate_mass,
    select_representative,
)


def _make_physical_state(
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: np.ndarray | None = None,
    points: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    boundary: int = 0,
    grip: float = 0.0,
) -> PhysicalState:
    pose = np.eye(4, dtype=np.float32)
    if rotation is not None:
        pose[:3, :3] = rotation.astype(np.float32)
    pose[:3, 3] = np.asarray(translation, dtype=np.float32)

    if points is None:
        pts = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]],
            dtype=torch.float32,
        )
    else:
        pts = torch.as_tensor(points, dtype=torch.float32)
        if pts.ndim == 2:
            pts = pts.unsqueeze(0)

    if valid_mask is None:
        vmask = torch.ones(pts.shape[:2], dtype=torch.bool)
    else:
        vmask = torch.as_tensor(valid_mask, dtype=torch.bool)
        if vmask.ndim == 1:
            vmask = vmask.unsqueeze(0)

    anchor_valid = torch.zeros(1, 128, dtype=torch.bool)
    anchor_valid[:, : pts.shape[1]] = True

    return PhysicalState(
        X=torch.ones(1, 128, 256, dtype=torch.float32),
        x=torch.zeros(1, 128, 3, dtype=torch.float32),
        valid=anchor_valid,
        p=torch.zeros(1, 13, dtype=torch.float32),
        memory=torch.zeros(1, 2, 256, dtype=torch.float32),
        T_w_e=torch.from_numpy(pose).unsqueeze(0),
        grip=torch.tensor([[grip]], dtype=torch.float32),
        cached_world_cloud=pts,
        cached_world_cloud_valid=vmask,
        boundary=boundary,
        encoder_lineage="encoder-test",
        memory_lineage="memory-test",
        origin="imagined",
    )


class SearchBeliefTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = MethodConfig()

    def test_step_1_mass_conservation_and_deadline_return(self) -> None:
        """Step 1 assertion body from task-1-brief.md with explicit config."""
        u, f, w = propagate_mass(
            0.1,
            0.0,
            np.array([0.3, 0.3, 0.3]),
            np.array([[0.2, 0.1, 0.7]] * 3),
            self.cfg,
        )
        self.assertAlmostEqual(u + f + w.sum(), 1.0)
        self.assertAlmostEqual(leaf_return(u, w, np.ones(3), 0), u)
        self.assertAlmostEqual(leaf_return(u, w, np.ones(3), 3), u + w.sum())

    def test_mass_propagation_validation_rejects_invalid_inputs(self) -> None:
        """Propagate mass rejects invalid probabilities, total mass mismatch, and negative weights."""
        valid_weights = np.array([0.3, 0.3, 0.3])
        valid_events = np.array([[0.2, 0.1, 0.7]] * 3)

        # Non-MethodConfig / NumericsConfig
        with self.assertRaises(TypeError):
            propagate_mass(0.1, 0.0, valid_weights, valid_events, None)  # type: ignore[arg-type]

        # Prior mass mismatch (0.2 + 0.0 + 0.9 = 1.1 != 1.0)
        with self.assertRaises(ValueError):
            propagate_mass(0.2, 0.0, valid_weights, valid_events, self.cfg)

        # Negative mass
        with self.assertRaises(ValueError):
            propagate_mass(-0.1, 0.2, np.array([0.3, 0.3, 0.3]), valid_events, self.cfg)
        with self.assertRaises(ValueError):
            propagate_mass(0.1, 0.0, np.array([-0.1, 0.5, 0.5]), valid_events, self.cfg)

        # Invalid event probabilities (does not sum to 1.0; never silently renormalize)
        invalid_events = np.array([[0.5, 0.5, 0.5], [0.2, 0.1, 0.7], [0.2, 0.1, 0.7]])
        with self.assertRaises(ValueError):
            propagate_mass(0.1, 0.0, valid_weights, invalid_events, self.cfg)

        # Shape mismatch
        with self.assertRaises(ValueError):
            propagate_mass(0.1, 0.0, np.array([0.45, 0.45]), valid_events, self.cfg)

    def test_zero_active_mass_means_exactly_zero(self) -> None:
        """When continuation probability is 0, active mass becomes exactly zero."""
        u, f, w = propagate_mass(
            0.1,
            0.0,
            np.array([0.3, 0.3, 0.3]),
            np.array([[0.5, 0.5, 0.0]] * 3),
            self.cfg,
        )
        self.assertAlmostEqual(u + f, 1.0)
        self.assertEqual(list(w), [0.0, 0.0, 0.0])
        self.assertTrue((w == 0.0).all())

    def test_leaf_return_validations_and_deadlines(self) -> None:
        """Leaf return validates remaining horizon and vector compatibility."""
        weights = np.array([0.2, 0.3, 0.4])
        values = np.array([0.5, 0.8, 0.2])

        # Remaining == 0 strictly returns U
        self.assertEqual(leaf_return(0.1, weights, values, 0), 0.1)

        # Remaining > 0 computes expected return
        expected = 0.1 + (weights * values).sum()
        self.assertAlmostEqual(leaf_return(0.1, weights, values, 5), expected)

        # Negative remaining rejected
        with self.assertRaises(ValueError):
            leaf_return(0.1, weights, values, -1)

        # Length mismatch rejected
        with self.assertRaises(ValueError):
            leaf_return(0.1, weights, np.array([0.5, 0.8]), 2)

    def test_hypothesis_and_belief_node_ownership_and_invariants(self) -> None:
        """Hypothesis and BeliefNode enforce head IDs, mass conservation, and state types."""
        s0 = _make_physical_state(translation=(0.0, 0.0, 0.0))
        s1 = _make_physical_state(translation=(0.1, 0.0, 0.0))
        s2 = _make_physical_state(translation=(0.2, 0.0, 0.0))

        # Invalid head_id
        with self.assertRaises(ValueError):
            Hypothesis(head_id=3, state=s0, task={"id": 0}, weight=0.3)
        with self.assertRaises(ValueError):
            Hypothesis(head_id=-1, state=s0, task={"id": 0}, weight=0.3)

        # Negative mass in Hypothesis
        with self.assertRaises(ValueError):
            Hypothesis(head_id=0, state=s0, task={"id": 0}, weight=-0.01)

        # Real PhysicalState requirement
        with self.assertRaises(TypeError):
            Hypothesis(head_id=0, state=object(), task={"id": 0}, weight=0.3)  # type: ignore[arg-type]

        h0 = Hypothesis(head_id=0, state=s0, task={"id": 0}, weight=0.3)
        h1 = Hypothesis(head_id=1, state=s1, task={"id": 1}, weight=0.3)
        h2 = Hypothesis(head_id=2, state=s2, task={"id": 2}, weight=0.3)

        # Valid BeliefNode
        node = BeliefNode(
            tau=0,
            H_root=10,
            U=0.1,
            F=0.0,
            hypotheses=(h0, h1, h2),
            cfg=self.cfg,
        )
        np.testing.assert_allclose(node.weights, [0.3, 0.3, 0.3])
        self.assertFalse(node.is_terminal)

        # tau > H_root rejected
        with self.assertRaises(ValueError):
            BeliefNode(
                tau=11,
                H_root=10,
                U=0.1,
                F=0.0,
                hypotheses=(h0, h1, h2),
                cfg=self.cfg,
            )

        # Non-conserved mass rejected (0.1 + 0.1 + 0.9 = 1.1 != 1.0)
        with self.assertRaises(ValueError):
            BeliefNode(
                tau=0,
                H_root=10,
                U=0.2,
                F=0.0,
                hypotheses=(h0, h1, h2),
                cfg=self.cfg,
            )

    def test_representative_selection_medoid_and_exact_tie_breaking(self) -> None:
        """Medoid minimizes pairwise scaled CD + translation + rotation, breaking exact ties to lower head."""
        # Head 1 is located centrally between Head 0 and Head 2
        s0 = _make_physical_state(translation=(-0.05, 0.0, 0.0))
        s1 = _make_physical_state(translation=(0.0, 0.0, 0.0))
        s2 = _make_physical_state(translation=(0.05, 0.0, 0.0))

        h0 = Hypothesis(head_id=0, state=s0, task=None, weight=1.0 / 3.0)
        h1 = Hypothesis(head_id=1, state=s1, task=None, weight=1.0 / 3.0)
        h2 = Hypothesis(head_id=2, state=s2, task=None, weight=1.0 / 3.0)

        rep = select_representative((h0, h1, h2), self.cfg)
        self.assertEqual(rep.head_id, 1)

        # Exact tie test: symmetric heads 0 and 2 when head 1 has 0 mass
        # Head 0 at (-0.02, 0, 0) and Head 2 at (0.02, 0, 0); distance between them is symmetric
        h0_sym = Hypothesis(head_id=0, state=s0, task=None, weight=0.5)
        h1_zero = Hypothesis(head_id=1, state=s1, task=None, weight=0.0)
        h2_sym = Hypothesis(head_id=2, state=s2, task=None, weight=0.5)

        # Distance d(0, 2) == d(2, 0). Exact tie must select lower head ID: head 0.
        rep_sym = select_representative((h0_sym, h1_zero, h2_sym), self.cfg)
        self.assertEqual(rep_sym.head_id, 0)

        # Single active hypothesis selected directly
        h0_only = Hypothesis(head_id=0, state=s0, task=None, weight=0.0)
        h1_only = Hypothesis(head_id=1, state=s1, task=None, weight=0.0)
        h2_only = Hypothesis(head_id=2, state=s2, task=None, weight=1.0)
        self.assertEqual(select_representative((h0_only, h1_only, h2_only), self.cfg).head_id, 2)

        # All zero active mass rejected
        h_zeros = (
            Hypothesis(head_id=0, state=s0, task=None, weight=0.0),
            Hypothesis(head_id=1, state=s1, task=None, weight=0.0),
            Hypothesis(head_id=2, state=s2, task=None, weight=0.0),
        )
        with self.assertRaises(ValueError):
            select_representative(h_zeros, self.cfg)

    def test_representative_selection_geometry_masks_and_empty_rejection(self) -> None:
        """Geometry masks are respected and empty geometry raises ValueError without silent zero distance."""
        pts = np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [10.0, 10.0, 10.0]],
            dtype=np.float32,
        )
        # Point 2 is an outlier at (10, 10, 10), masked out by valid_mask
        mask = np.array([True, True, False])

        s0 = _make_physical_state(points=pts, valid_mask=mask)
        s1 = _make_physical_state(points=pts, valid_mask=mask)
        s2 = _make_physical_state(points=pts, valid_mask=mask)

        h0 = Hypothesis(head_id=0, state=s0, task=None, weight=1.0 / 3.0)
        h1 = Hypothesis(head_id=1, state=s1, task=None, weight=1.0 / 3.0)
        h2 = Hypothesis(head_id=2, state=s2, task=None, weight=1.0 / 3.0)

        # All identical masked geometry: exact tie broken to head 0
        rep = select_representative((h0, h1, h2), self.cfg)
        self.assertEqual(rep.head_id, 0)

        # Empty valid cloud rejected
        empty_mask = np.array([False, False, False])
        s_empty = _make_physical_state(points=pts, valid_mask=empty_mask)
        h_empty = Hypothesis(head_id=0, state=s_empty, task=None, weight=1.0)
        with self.assertRaises(ValueError):
            select_representative((h_empty,), self.cfg)

    def test_belief_node_methods_and_precision_tolerances(self) -> None:
        """BeliefNode select_representative, terminal detection, and FP32/FP64 tolerance checks."""
        s0 = _make_physical_state(translation=(0.0, 0.0, 0.0))
        s1 = _make_physical_state(translation=(0.05, 0.0, 0.0))
        s2 = _make_physical_state(translation=(0.1, 0.0, 0.0))

        h0 = Hypothesis(head_id=0, state=s0, task=None, weight=0.4)
        h1 = Hypothesis(head_id=1, state=s1, task=None, weight=0.3)
        h2 = Hypothesis(head_id=2, state=s2, task=None, weight=0.2)

        node = BeliefNode(tau=0, H_root=5, U=0.1, F=0.0, hypotheses=(h0, h1, h2), cfg=self.cfg)
        rep = node.select_representative()
        self.assertEqual(rep.head_id, 1)
        self.assertFalse(node.is_terminal)

        # Terminal when tau == H_root
        node_deadline = BeliefNode(tau=5, H_root=5, U=0.1, F=0.0, hypotheses=(h0, h1, h2), cfg=self.cfg)
        self.assertTrue(node_deadline.is_terminal)

        # Terminal when all weights are zero
        h0_z = Hypothesis(head_id=0, state=s0, task=None, weight=0.0)
        h1_z = Hypothesis(head_id=1, state=s1, task=None, weight=0.0)
        h2_z = Hypothesis(head_id=2, state=s2, task=None, weight=0.0)
        node_zero_active = BeliefNode(tau=1, H_root=5, U=0.6, F=0.4, hypotheses=(h0_z, h1_z, h2_z), cfg=self.cfg)
        self.assertTrue(node_zero_active.is_terminal)

        # FP32 mass tolerance: 1e-5 allowed in FP32, but 1e-6 drift rejected in FP64
        w_fp32 = np.array([0.3333333, 0.3333333, 0.3333333], dtype=np.float32)
        ev_fp32 = np.array([[0.3333333, 0.3333333, 0.3333334]] * 3, dtype=np.float32)
        # sum is 0.9999999, diff is 1e-7 <= 1e-5
        u_fp32, f_fp32, next_w_fp32 = propagate_mass(0.0000001, 0.0, w_fp32, ev_fp32, self.cfg)
        self.assertEqual(next_w_fp32.dtype, np.float32)

        # FP64 strict tolerance (1e-12): drift of 1e-9 must be rejected
        w_fp64 = np.array([0.3, 0.3, 0.3], dtype=np.float64)
        ev_fp64_bad = np.array([[0.333333333333, 0.333333333333, 0.333333333335]] * 3, dtype=np.float64)
        with self.assertRaises(ValueError):
            propagate_mass(0.1, 0.0, w_fp64, ev_fp64_bad, self.cfg)

    def test_regression_strict_probability_bounds(self) -> None:
        """Probabilities strictly outside [0, 1] must be rejected even if within float32 sum atol."""
        weights = np.array([0.3, 0.3, 0.4], dtype=np.float32)
        # Element strictly > 1.0 (e.g. 1.000005) must fail
        events_over = np.array([
            [1.000005, 0.0, 0.0],
            [0.2, 0.1, 0.7],
            [0.2, 0.1, 0.7],
        ], dtype=np.float32)
        with self.assertRaises(ValueError):
            propagate_mass(0.0, 0.0, weights, events_over, self.cfg)

        # Element strictly < 0.0 must fail
        events_under = np.array([
            [-0.000005, 0.5, 0.5],
            [0.2, 0.1, 0.7],
            [0.2, 0.1, 0.7],
        ], dtype=np.float32)
        with self.assertRaises(ValueError):
            propagate_mass(0.0, 0.0, weights, events_under, self.cfg)

    def test_regression_dtype_continuity_propagate_hypothesis_belief_node(self) -> None:
        """Mass arithmetic dtype must be preserved through propagate -> Hypothesis -> BeliefNode."""
        s0 = _make_physical_state(translation=(0.0, 0.0, 0.0))
        s1 = _make_physical_state(translation=(0.01, 0.0, 0.0))
        s2 = _make_physical_state(translation=(0.02, 0.0, 0.0))

        w_fp32 = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        ev_fp32 = np.array([
            [0.2, 0.1, 0.7],
            [0.2, 0.1, 0.7],
            [0.2, 0.1, 0.7],
        ], dtype=np.float32)
        u, f, w = propagate_mass(0.1, 0.0, w_fp32, ev_fp32, self.cfg)

        h0 = Hypothesis(head_id=0, state=s0, task=None, weight=w[0])
        h1 = Hypothesis(head_id=1, state=s1, task=None, weight=w[1])
        h2 = Hypothesis(head_id=2, state=s2, task=None, weight=w[2])

        node = BeliefNode(tau=0, H_root=5, U=u, F=f, hypotheses=(h0, h1, h2), cfg=self.cfg)
        self.assertEqual(node.weights.dtype, np.float32)

        # Drift of 2e-7 in FP32 must be accepted under atol=1e-5
        w_drift_fp32 = np.array([0.3333333, 0.3333333, 0.3333335], dtype=np.float32)
        h0_d = Hypothesis(head_id=0, state=s0, task=None, weight=w_drift_fp32[0])
        h1_d = Hypothesis(head_id=1, state=s1, task=None, weight=w_drift_fp32[1])
        h2_d = Hypothesis(head_id=2, state=s2, task=None, weight=w_drift_fp32[2])
        node_fp32 = BeliefNode(tau=0, H_root=5, U=0.0, F=0.0, hypotheses=(h0_d, h1_d, h2_d), cfg=self.cfg)
        self.assertEqual(node_fp32.weights.dtype, np.float32)

        # Drift of 2e-7 in FP64 must be rejected under atol=1e-12
        w_drift_fp64 = np.array([0.3333333, 0.3333333, 0.3333335], dtype=np.float64)
        h0_64 = Hypothesis(head_id=0, state=s0, task=None, weight=w_drift_fp64[0])
        h1_64 = Hypothesis(head_id=1, state=s1, task=None, weight=w_drift_fp64[1])
        h2_64 = Hypothesis(head_id=2, state=s2, task=None, weight=w_drift_fp64[2])
        with self.assertRaises(ValueError):
            BeliefNode(tau=0, H_root=5, U=0.0, F=0.0, hypotheses=(h0_64, h1_64, h2_64), cfg=self.cfg)

    def test_regression_mixed_fp32_fp64_dtype_consistency_and_no_downcast(self) -> None:
        """Mixed FP32 and FP64 inputs must promote via np.result_type to FP64 without downcasting drift."""
        s0 = _make_physical_state(translation=(0.0, 0.0, 0.0))
        s1 = _make_physical_state(translation=(0.01, 0.0, 0.0))
        s2 = _make_physical_state(translation=(0.02, 0.0, 0.0))

        # Mixed inputs: weights FP64, events FP32
        w64 = np.array([0.3, 0.3, 0.3], dtype=np.float64)
        ev32 = np.array([
            [0.25, 0.25, 0.5],
            [0.25, 0.25, 0.5],
            [0.25, 0.25, 0.5],
        ], dtype=np.float32)

        u, f, w = propagate_mass(0.1, 0.0, w64, ev32, self.cfg)
        # All outputs must be consistent FP64 (promoted by np.result_type)
        self.assertEqual(getattr(u, "dtype", None), np.float64)
        self.assertEqual(getattr(f, "dtype", None), np.float64)
        self.assertEqual(w.dtype, np.float64)

        # Mixed inputs reversed: weights FP32, events FP64
        w32 = np.array([0.25, 0.25, 0.25], dtype=np.float32)
        ev64 = np.array([
            [0.25, 0.25, 0.5],
            [0.25, 0.25, 0.5],
            [0.25, 0.25, 0.5],
        ], dtype=np.float64)
        u_rev, f_rev, w_rev = propagate_mass(0.25, 0.0, w32, ev64, self.cfg)
        self.assertEqual(getattr(u_rev, "dtype", None), np.float64)
        self.assertEqual(getattr(f_rev, "dtype", None), np.float64)
        self.assertEqual(w_rev.dtype, np.float64)

        # Mixed node: FP64 weights with FP64 drift of 2e-7 must NOT be accepted under FP32 tolerance
        # even if U or another operand was FP32
        u_fp32 = np.float32(0.0)
        w_drift_64 = np.array([0.3333333, 0.3333333, 0.3333335], dtype=np.float64)
        h0_64 = Hypothesis(head_id=0, state=s0, task=None, weight=w_drift_64[0])
        h1_64 = Hypothesis(head_id=1, state=s1, task=None, weight=w_drift_64[1])
        h2_64 = Hypothesis(head_id=2, state=s2, task=None, weight=w_drift_64[2])
        with self.assertRaises(ValueError):
            BeliefNode(tau=0, H_root=5, U=u_fp32, F=0.0, hypotheses=(h0_64, h1_64, h2_64), cfg=self.cfg)

        # When node is created from mixed inputs without drift, node.weights must be FP64 (not downcast)
        w_nodrift_64 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        h0_ok = Hypothesis(head_id=0, state=s0, task=None, weight=w_nodrift_64[0])
        h1_ok = Hypothesis(head_id=1, state=s1, task=None, weight=w_nodrift_64[1])
        h2_ok = Hypothesis(head_id=2, state=s2, task=None, weight=w_nodrift_64[2])
        node_mixed = BeliefNode(tau=0, H_root=5, U=u_fp32, F=0.0, hypotheses=(h0_ok, h1_ok, h2_ok), cfg=self.cfg)
        self.assertEqual(node_mixed.weights.dtype, np.float64)

    def test_regression_medoid_rotation_small_angles_and_pi(self) -> None:
        """Medoid rotation metric clamps [-1, 1] without training margin, selecting central small rotation."""
        import math

        def _rot_z(theta: float) -> np.ndarray:
            c, s = math.cos(theta), math.sin(theta)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        s0 = _make_physical_state(rotation=_rot_z(0.0))
        s1 = _make_physical_state(rotation=_rot_z(0.0005))
        s2 = _make_physical_state(rotation=_rot_z(0.001))

        h0 = Hypothesis(head_id=0, state=s0, task=None, weight=1.0 / 3.0)
        h1 = Hypothesis(head_id=1, state=s1, task=None, weight=1.0 / 3.0)
        h2 = Hypothesis(head_id=2, state=s2, task=None, weight=1.0 / 3.0)

        # Head 1 is strictly central: medoid must select head 1
        rep = select_representative((h0, h1, h2), self.cfg)
        self.assertEqual(rep.head_id, 1)

    def test_regression_belief_node_h_root_bound(self) -> None:
        """BeliefNode H_root cannot exceed cfg.planning.H."""
        s0 = _make_physical_state()
        s1 = _make_physical_state()
        s2 = _make_physical_state()
        h0 = Hypothesis(head_id=0, state=s0, task=None, weight=1.0 / 3.0)
        h1 = Hypothesis(head_id=1, state=s1, task=None, weight=1.0 / 3.0)
        h2 = Hypothesis(head_id=2, state=s2, task=None, weight=1.0 / 3.0)

        # H_root == cfg.planning.H is allowed
        node_valid = BeliefNode(tau=0, H_root=self.cfg.planning.H, U=0.0, F=0.0, hypotheses=(h0, h1, h2), cfg=self.cfg)
        self.assertEqual(node_valid.H_root, self.cfg.planning.H)

        # H_root > cfg.planning.H must be rejected
        with self.assertRaises(ValueError):
            BeliefNode(tau=0, H_root=self.cfg.planning.H + 1, U=0.0, F=0.0, hypotheses=(h0, h1, h2), cfg=self.cfg)


def _make_timed_command(
    offset: float = 0.0,
    grip: int = 1,
    duration_s: float = 0.1,
) -> TimedCommand:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = offset
    return TimedCommand(target_w=pose, grip=grip, duration_s=duration_s)


def _make_evaluation_output(
    value: float = 0.5,
    stop: float = 0.1,
    horizon: int = 1,
) -> EvaluationOutput:
    val_arr = np.array([0.0 if horizon == 0 else value], dtype=np.float64)
    stop_arr = np.array([stop], dtype=np.float64)
    logits = np.zeros(1, dtype=np.float64)
    return EvaluationOutput(
        value_logits=logits,
        completion_logits=logits,
        progress_logits=logits,
        calibrated_value=val_arr,
        calibrated_stop=stop_arr,
        temperatures_id="temp_artifact_v1",
        horizon=horizon,
    )


def _make_terminal_probabilities(
    probs: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> TerminalProbabilities:
    return TerminalProbabilities(
        probabilities=np.array([probs], dtype=np.float64),
        event_temperature_artifact_id="event_temp_v1",
    )


def _make_prediction(state: PhysicalState, head_id: int) -> PhysicalPrediction:
    return PhysicalPrediction(
        next_state=state,
        grip_logits=np.zeros((1, 1), dtype=np.float32),
        head_id=head_id,
    )


class DeterministicMockCapabilities:
    """Deterministic collaborator implementing MethodCapabilities protocol for testing."""

    def __init__(
        self,
        *,
        step_terminal_probs: tuple[float, float, float] = (0.0, 0.0, 1.0),
        head_values: dict[int, float] | None = None,
        candidate_commands: list[tuple[TimedCommand, ...]] | None = None,
        world_model_id: str = "mock_world_model_v1",
        task_tracker_id: str = "mock_tracker_v1",
        terminal_id: str = "mock_terminal_v1",
        stop_value: float = 0.1,
    ) -> None:
        self.step_terminal_probs = step_terminal_probs
        self.head_values = head_values or {0: 0.4, 1: 0.5, 2: 0.6}
        self.candidate_commands = candidate_commands or []
        self.world_model_id = world_model_id
        self.task_tracker_id = task_tracker_id
        self.terminal_id = terminal_id
        self.stop_value = stop_value

        self.sample_prior_calls: list[dict[str, Any]] = []
        self.materialize_calls: list[dict[str, Any]] = []
        self.predict_step_calls: list[dict[str, Any]] = []
        self.track_task_calls: list[dict[str, Any]] = []
        self.predict_terminal_calls: list[dict[str, Any]] = []
        self.evaluate_state_calls: list[dict[str, Any]] = []
        self.sync_calls = 0

    def synchronize(self) -> None:
        self.sync_calls += 1

    def validate_context(self, context: Any) -> None:
        if not isinstance(context, dict):
            raise TypeError("context must be a mapping")
        if not context.get("context_id"):
            raise ValueError("context must contain nonempty context_id")

    def sample_prior(self, observation: Any, task: Any, context: Any, *, seed: int) -> Any:
        idx = len(self.sample_prior_calls)
        call_info = {"idx": idx, "seed": seed, "obs": observation, "task": task, "context": context}
        self.sample_prior_calls.append(call_info)
        return {"index": idx, "seed": seed, "raw_candidate_id": f"cand_{idx}"}

    def materialize_prefix(self, candidate: Any, *, h: int, r: int, duration_s: float) -> CommandPrefix:
        self.materialize_calls.append({"candidate": candidate, "h": h, "r": r, "duration_s": duration_s})
        cand_idx = candidate["index"] if isinstance(candidate, dict) else 0
        if self.candidate_commands and cand_idx < len(self.candidate_commands):
            cmds = self.candidate_commands[cand_idx][:h]
            if len(cmds) < h:
                cmds = cmds + tuple(
                    _make_timed_command(offset=0.01 * (cand_idx + 1) + 0.001 * step, duration_s=duration_s)
                    for step in range(len(cmds), h)
                )
        else:
            cmds = tuple(
                _make_timed_command(offset=0.01 * (cand_idx + 1) + 0.001 * step, duration_s=duration_s)
                for step in range(h)
            )
        root_pose = np.eye(4, dtype=np.float64)
        return CommandPrefix(
            commands=cmds,
            proposal_root=root_pose,
            raw_candidate_id=str(candidate.get("raw_candidate_id", f"cand_{cand_idx}")),
        )

    def predict_step(self, state: PhysicalState, command: TimedCommand, *, head_id: int) -> PhysicalPrediction:
        self.predict_step_calls.append({"state_boundary": state.boundary, "command": command, "head_id": head_id})
        next_state = _make_physical_state(
            translation=(float(command.target_w[0, 3]), 0.0, 0.0),
            boundary=state.boundary + 1,
        )
        return _make_prediction(next_state, head_id)

    def track_task(self, previous: Any, state: PhysicalState, events: Any) -> Any:
        self.track_task_calls.append({"previous": previous, "boundary": state.boundary})
        history_id = previous.get("history_id", "h0") if isinstance(previous, dict) else "h0"
        return {"step": state.boundary, "boundary": state.boundary, "history_id": f"{history_id}_b{state.boundary}"}

    def predict_terminal(
        self,
        before: PhysicalState,
        q_before: Any,
        after: PhysicalState,
        q_after: Any,
        events: Any,
        command: TimedCommand,
    ) -> TerminalProbabilities:
        self.predict_terminal_calls.append({"before": before.boundary, "after": after.boundary})
        return _make_terminal_probabilities(self.step_terminal_probs)

    def evaluate_state(self, state: PhysicalState, task: Any, events: Any, H: int) -> EvaluationOutput:
        self.evaluate_state_calls.append({"boundary": state.boundary, "H": H})
        val = 0.5
        return _make_evaluation_output(value=val, stop=self.stop_value, horizon=H)


MockCapabilities = DeterministicMockCapabilities


class TestRecorder:
    """Recording spy implementing observability recorder seam for testing."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def event(
        self,
        name: str,
        *,
        level: str = "INFO",
        component: str | None = None,
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append({
            "name": name,
            "level": level,
            "component": component,
            "fields": dict(fields or {}),
        })

    def find_events(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["name"] == name]


class SearchMCTSTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = MethodConfig()

    def test_step_1_progressive_widening_and_uct(self) -> None:
        """Step 1 assertion body from task-2-brief.md with explicit config."""
        from icgs.algorithms.planning.mcts import widening_limit, uct
        self.assertEqual(widening_limit(0, self.cfg), 1)
        self.assertEqual(widening_limit(3, self.cfg), 3)
        with self.assertRaisesRegex(ValueError, 'unvisited'):
            uct(0., 0, 1, self.cfg)

    def test_widening_limit_and_uct_validations_and_config(self) -> None:
        """Widening limit and UCT validate inputs and respect explicit config tuning."""
        from icgs.algorithms.planning.mcts import widening_limit, uct

        custom_cfg = MethodConfig(
            planning={
                "widening_coefficient": 2.0,
                "widening_exponent": 1.0,
                "uct_exploration": 2.0,
            }
        )
        self.assertEqual(widening_limit(2, custom_cfg), 6)
        expected_uct = 1.0 + 2.0 * math.sqrt(math.log(5.0) / 2.0)
        self.assertAlmostEqual(uct(2.0, 2, 4, custom_cfg), expected_uct)

        # Negative visits rejected
        with self.assertRaises(ValueError):
            widening_limit(-1, self.cfg)
        with self.assertRaises(ValueError):
            uct(1.0, -1, 2, self.cfg)
        with self.assertRaises(ValueError):
            uct(1.0, 2, -1, self.cfg)

        # Non-numeric types rejected
        with self.assertRaises(TypeError):
            widening_limit("zero", self.cfg)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            uct("one", 1, 1, self.cfg)  # type: ignore[arg-type]

        # Non-finite returns rejected
        with self.assertRaises(ValueError):
            uct(float("nan"), 1, 1, self.cfg)
        with self.assertRaises(ValueError):
            uct(float("inf"), 1, 1, self.cfg)

    def test_hand_enumerated_two_depth_tree_visits_and_backup(self) -> None:
        """Hand-enumerated 3-iteration search generates a 2-depth tree with exact visits and one G backup."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan

        caps = DeterministicMockCapabilities()
        state = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        rec = TestRecorder()

        # Bounded by native_call_cap=3 (3 expansions)
        budget = PlanningBudget(native_call_cap=3)
        result = plan(
            state,
            task,
            context,
            H=4,
            budget=budget,
            capabilities=caps,
            cfg=self.cfg,
            recorder=rec,
        )

        self.assertTrue(result.completed)
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(result.call_count, 3)
        self.assertEqual(result.timing_counters["iterations"], 3)
        self.assertEqual(result.timing_counters["native_calls"], 3)

        # Verify node visits from recorder events
        node_events = rec.find_events("mcts.node_visit")
        # Root visited in all 3 iterations
        root_visits = [e for e in node_events if e["fields"]["tau"] == 0]
        self.assertEqual(len(root_visits), 3)
        self.assertEqual(root_visits[-1]["fields"]["visits"], 3)

        # Verify edge backups
        edge_events = rec.find_events("mcts.edge_backup")
        # Edge 0 visited in iteration 1 and iteration 3 -> final visits 2
        edge0_backups = [e for e in edge_events if e["fields"]["edge_id"] == 0]
        self.assertEqual(len(edge0_backups), 2)
        self.assertEqual(edge0_backups[-1]["fields"]["visits"], 2)
        self.assertAlmostEqual(edge0_backups[-1]["fields"]["total_return"], 0.65)

        # Edge 1 visited in iteration 2 -> final visits 1
        edge1_backups = [e for e in edge_events if e["fields"]["edge_id"] == 1]
        self.assertEqual(len(edge1_backups), 1)
        self.assertEqual(edge1_backups[-1]["fields"]["visits"], 1)
        self.assertAlmostEqual(edge1_backups[-1]["fields"]["total_return"], 0.55)

        # Depth-2 edge (Edge 2) expanded and visited in iteration 3 -> final visits 1
        edge2_backups = [e for e in edge_events if e["fields"]["edge_id"] == 2]
        self.assertEqual(len(edge2_backups), 1)
        self.assertEqual(edge2_backups[0]["fields"]["visits"], 1)
        self.assertAlmostEqual(edge2_backups[0]["fields"]["total_return"], 0.1)

        # Depth-1 child nodes at tau=2: Edge 0 child visited in iter 1 and 3 (final visits 2), Edge 1 child in iter 2 (visits 1)
        child_visits = [e for e in node_events if e["fields"]["tau"] == 2]
        self.assertEqual(len(child_visits), 3)
        edge0_child_final = [e for e in child_visits if e["fields"]["visits"] == 2]
        self.assertEqual(len(edge0_child_final), 1)

        # Depth-2 leaf node at tau=4 visited in iteration 3 -> visits 1
        leaf_visits = [e for e in node_events if e["fields"]["tau"] == 4]
        self.assertEqual(len(leaf_visits), 1)
        self.assertEqual(leaf_visits[0]["fields"]["visits"], 1)

        # Edge 0 selected because it has 2 visits vs Edge 1's 1 visit
        self.assertIsNotNone(result.selected_prefix)
        self.assertAlmostEqual(result.expected_return, 0.325)

    def test_exact_cache_lineage_and_task_history_separation(self) -> None:
        """Exact cache reuses identical transitions and enforces task-history / time separation without fake visits."""
        from icgs.algorithms.planning.mcts import CacheKey, CompositeModelIdentity, ExactCache

        caps = DeterministicMockCapabilities()
        cache = ExactCache("root_test")
        models = CompositeModelIdentity("wm_1", "tt_1", "term_1")
        key1 = CacheKey(
            parent_state_digest=b"digest_state_a",
            task_identity="task_h1",
            context_identity="ctx_0",
            head_id=0,
            composite_models=models,
            tau=0,
            command_bytes=b"cmd_bytes_1",
        )
        dummy_state = _make_physical_state(boundary=1)
        dummy_probs = np.array([0.1, 0.0, 0.9])
        transition = (dummy_state, {"history_id": "task_h1_b1"}, dummy_probs)

        # Initial miss
        self.assertIsNone(cache.get(key1, caps))
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.lookups, 1)

        # Store in cache
        cache.put(key1, transition)

        # Cache hit with exact lineage
        hit_result = cache.get(key1, caps)
        self.assertIsNotNone(hit_result)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.lookups, 2)

        # Task history separation: different task lineage must miss
        key_diff_task = CacheKey(
            parent_state_digest=b"digest_state_a",
            task_identity="task_DIFF_HISTORY",
            context_identity="ctx_0",
            head_id=0,
            composite_models=models,
            tau=0,
            command_bytes=b"cmd_bytes_1",
        )
        self.assertIsNone(cache.get(key_diff_task, caps))
        self.assertEqual(cache.misses, 2)

        # Time / tau separation: different tau must miss
        key_diff_tau = CacheKey(
            parent_state_digest=b"digest_state_a",
            task_identity="task_h1",
            context_identity="ctx_0",
            head_id=0,
            composite_models=models,
            tau=1,
            command_bytes=b"cmd_bytes_1",
        )
        self.assertIsNone(cache.get(key_diff_tau, caps))
        self.assertEqual(cache.misses, 3)

    def test_exact_cache_root_separation(self) -> None:
        """Exact cache is root-scoped; transitions from one root are never carried to another root."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan

        caps = DeterministicMockCapabilities()
        state = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        # Run root A
        res_a = plan(state, task, context, H=4, budget=PlanningBudget(native_call_cap=1), capabilities=caps, cfg=self.cfg)
        # Run root B with fresh execution
        res_b = plan(state, task, context, H=4, budget=PlanningBudget(native_call_cap=1), capabilities=caps, cfg=self.cfg)

        # Each root executed its own search and counters
        self.assertEqual(res_a.cache_counters["misses"], 2 * 3)  # 2 intervals * 3 heads
        self.assertEqual(res_b.cache_counters["misses"], 2 * 3)
        self.assertEqual(res_a.cache_counters["hits"], 0)
        self.assertEqual(res_b.cache_counters["hits"], 0)

    def test_materialize_once_and_command_equality_across_all_heads(self) -> None:
        """Candidate prefix is materialized ONCE and applied identically across all 3 heads."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan

        caps = DeterministicMockCapabilities()
        state = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        # Run 1 iteration -> expands 1 edge (2 intervals)
        plan(state, task, context, H=4, budget=PlanningBudget(native_call_cap=1), capabilities=caps, cfg=self.cfg)

        # materialize_prefix called exactly once
        self.assertEqual(len(caps.materialize_calls), 1)

        # predict_step called for 2 intervals * 3 heads = 6 calls
        self.assertEqual(len(caps.predict_step_calls), 6)

        # Verify commands for heads 0, 1, 2 in step 0 are byte-identical
        cmd_head0_step0 = caps.predict_step_calls[0]["command"]
        cmd_head1_step0 = caps.predict_step_calls[1]["command"]
        cmd_head2_step0 = caps.predict_step_calls[2]["command"]

        np.testing.assert_array_equal(cmd_head0_step0.target_w, cmd_head1_step0.target_w)
        np.testing.assert_array_equal(cmd_head0_step0.target_w, cmd_head2_step0.target_w)
        self.assertEqual(cmd_head0_step0.grip, cmd_head1_step0.grip)
        self.assertEqual(cmd_head0_step0.grip, cmd_head2_step0.grip)
        self.assertEqual(cmd_head0_step0.duration_s, cmd_head1_step0.duration_s)
        self.assertEqual(cmd_head0_step0.duration_s, cmd_head2_step0.duration_s)

    def test_deterministic_selection_and_final_choice_tie_breaking(self) -> None:
        """Deterministic tie breaking: earlier insertion ID for selection ties; visits then Q then earlier ID for final choice."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan

        caps = DeterministicMockCapabilities()
        state = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        rec = TestRecorder()

        # Run 3 native calls:
        # Iteration 1: Edge 0 (visits 1, Q 0.55)
        # Iteration 2: Edge 1 (visits 1, Q 0.55)
        # Iteration 3: Selection tie! Both Edge 0 and Edge 1 have equal UCT -> must select Edge 0
        budget = PlanningBudget(native_call_cap=3)
        res = plan(state, task, context, H=4, budget=budget, capabilities=caps, cfg=self.cfg, recorder=rec)

        # Check selection tie broke to edge 0
        selections = rec.find_events("mcts.selection")
        self.assertGreaterEqual(len(selections), 1)
        self.assertEqual(selections[0]["fields"]["selected_edge_id"], 0)

        # Case 1: Root choice broke to Edge 0 (most visits: 2 > 1)
        self.assertEqual(res.selected_prefix.raw_candidate_id, "cand_0")

        # Case 2: Equal visits (1 each), greater Q -> chooses edge with greater Q
        class CustomEvalCapabilities(DeterministicMockCapabilities):
            def evaluate_state(self, state: PhysicalState, task: Any, events: Any, H: int) -> EvaluationOutput:
                # Leaf of Edge 0 (boundary 2) gets value 0.2; Leaf of Edge 1 (boundary 4) gets value 0.8
                # But here boundary of Edge 0 is 2, and Edge 1 is also 2.
                # Differentiate by command offset or head_id or call count:
                eval_idx = len(self.evaluate_state_calls)
                self.evaluate_state_calls.append({"boundary": state.boundary, "H": H})
                # Root eval is idx 0. Iteration 1 eval is idx 1 (Edge 0). Iteration 2 eval is idx 2 (Edge 1).
                val = 0.2 if eval_idx <= 3 else 0.8
                return _make_evaluation_output(value=val, stop=self.stop_value, horizon=H)

        caps_q = CustomEvalCapabilities()
        res_q = plan(state, task, context, H=4, budget=PlanningBudget(native_call_cap=2), capabilities=caps_q, cfg=self.cfg)
        # Both edges visited once; Edge 1 has higher Q (0.8 vs 0.2) -> chooses cand_1
        self.assertEqual(res_q.selected_prefix.raw_candidate_id, "cand_1")

        # Case 3: Equal visits (1 each), equal Q -> chooses earlier edge ID (cand_0 over cand_1)
        caps_tie = DeterministicMockCapabilities()
        res_tie = plan(state, task, context, H=4, budget=PlanningBudget(native_call_cap=2), capabilities=caps_tie, cfg=self.cfg)
        # Both edges visited once, identical Q -> breaks tie to cand_0
        self.assertEqual(res_tie.selected_prefix.raw_candidate_id, "cand_0")

    def test_partial_edges_and_deadline_truncation(self) -> None:
        """When remaining horizon H - tau < h, edge is truncated to a partial edge carrying actual remaining steps."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan

        caps = DeterministicMockCapabilities()
        state = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        # Case 1: H=1 with h=2 at root -> partial edge with h=1
        res_root_partial = plan(
            state,
            task,
            context,
            H=1,
            budget=PlanningBudget(native_call_cap=1),
            capabilities=caps,
            cfg=self.cfg,
        )
        self.assertTrue(res_root_partial.completed)
        self.assertEqual(caps.materialize_calls[0]["h"], 1)

        # Case 2: H=3 with h=2:
        # Iteration 1: Edge 0 (h=2)
        # Iteration 2: Edge 1 (h=2)
        # Iteration 3: Child 0 has tau=2, rem=1 -> partial edge with h=1
        caps_deep = DeterministicMockCapabilities()
        res_deep = plan(
            state,
            task,
            context,
            H=3,
            budget=PlanningBudget(native_call_cap=3),
            capabilities=caps_deep,
            cfg=self.cfg,
        )
        self.assertTrue(res_deep.completed)
        self.assertEqual(caps_deep.materialize_calls[0]["h"], 2)
        self.assertEqual(caps_deep.materialize_calls[1]["h"], 2)
        self.assertEqual(caps_deep.materialize_calls[2]["h"], 1)

    def test_duplicate_sample_audit(self) -> None:
        """Duplicate candidate proposals are preserved in the audit log and flagged."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan

        identical_cmd = (_make_timed_command(offset=0.05), _make_timed_command(offset=0.06))
        caps = DeterministicMockCapabilities(
            candidate_commands=[identical_cmd, identical_cmd, identical_cmd]
        )
        state = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        rec = TestRecorder()

        result = plan(
            state,
            task,
            context,
            H=4,
            budget=PlanningBudget(native_call_cap=2),
            capabilities=caps,
            cfg=self.cfg,
            recorder=rec,
        )

        self.assertTrue(result.completed)
        self.assertEqual(len(result.audit_ids), 2)
        # Verify second sample hit cache for identical transition commands
        self.assertGreater(result.cache_counters["hits"], 0)

        # Verify duplicate flag and ordering via recorder
        audit_events = rec.find_events("candidate.audit")
        self.assertEqual(len(audit_events), 2)
        self.assertFalse(audit_events[0]["fields"]["is_duplicate"])
        self.assertEqual(audit_events[0]["fields"]["order"], 0)
        self.assertTrue(audit_events[1]["fields"]["is_duplicate"])
        self.assertEqual(audit_events[1]["fields"]["order"], 1)
        self.assertEqual(
            audit_events[0]["fields"]["commands_hash"],
            audit_events[1]["fields"]["commands_hash"],
        )

    def test_planning_budget_and_caps(self) -> None:
        """PlanningBudget enforces call caps and populates PlanningResult properly."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan

        caps = DeterministicMockCapabilities()
        state = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        budget = PlanningBudget(native_call_cap=2)
        result = plan(
            state,
            task,
            context,
            H=4,
            budget=budget,
            capabilities=caps,
            cfg=self.cfg,
        )

        self.assertEqual(result.call_count, 2)
        self.assertEqual(result.timing_counters["iterations"], 2)
        self.assertEqual(result.timing_counters["native_calls"], 2)
        self.assertTrue(result.completed)

    def test_audit_issue_1_exact_parent_identity_tensors_masks_grip_and_branch_copy(self) -> None:
        """Exact parent identity in cache key must differentiate states that differ only in physical tensors like points or grip."""
        from icgs.algorithms.planning.mcts import CacheKey, CompositeModelIdentity, ExactCache, _physical_state_identity

        caps = DeterministicMockCapabilities()
        state_a = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0)
        points_b = np.ones((1, 128, 3), dtype=np.float32) * 5.0
        state_b = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0, points=points_b)

        ident_a = _physical_state_identity(state_a)
        ident_b = _physical_state_identity(state_b)
        self.assertNotEqual(ident_a, ident_b)

        # Grip difference also differentiates
        state_grip1 = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0, grip=1.0)
        ident_grip1 = _physical_state_identity(state_grip1)
        self.assertNotEqual(ident_a, ident_grip1)

        # Verify branch copy isolation in ExactCache
        cache = ExactCache("root_test")
        models = CompositeModelIdentity("wm_1", "tt_1", "term_1")
        key = CacheKey(
            parent_state_digest=ident_a,
            task_identity="task_0",
            context_identity="ctx_0",
            head_id=0,
            composite_models=models,
            tau=0,
            command_bytes=b"cmd",
        )
        dummy_state = _make_physical_state(boundary=1)
        cache.put(key, (dummy_state, {"step": 1}, np.array([0.0, 0.0, 1.0])))

        cached_1 = cache.get(key, caps)
        self.assertIsNotNone(cached_1)
        # Mutate retrieved state
        cached_1[0].x[0, 0, 0] = 999.0

        # Retrieve again - must NOT have the mutation
        cached_2 = cache.get(key, caps)
        self.assertNotEqual(float(cached_2[0].x[0, 0, 0].item()), 999.0)

        # Verify non-dict mutable task child/cache isolation
        class MutableTask:
            def __init__(self, history_id: str, step: int) -> None:
                self.history_id = history_id
                self.step = step

            def branch_copy(self) -> MutableTask:
                import copy
                return copy.deepcopy(self)

        task_obj = MutableTask("task_custom", step=10)
        cache.put(key, (dummy_state, task_obj, np.array([0.0, 0.0, 1.0])))
        task_obj.step = 999
        retrieved_tk = cache.get(key, caps)[1]
        self.assertEqual(retrieved_tk.step, 10)

    def test_audit_issue_2_bounded_budget_and_operation_level_cap_interrupt(self) -> None:
        """PlanningBudget requires at least one bound; cap reached during edge stepping interrupts without evaluating incomplete edge."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan

        # Budget requires wall or native_call bound; model cap alone is insufficient
        with self.assertRaises(ValueError):
            PlanningBudget()
        with self.assertRaises(ValueError):
            PlanningBudget(model_interval_cap=1)

        caps = DeterministicMockCapabilities()
        state = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        # Edge needs 2 steps * 3 heads = 6 model intervals.
        # Cap at 1 model interval with wall budget: first predict_step happens, then cap reached.
        # Incomplete edge must NOT be evaluated, NOT added to tree, NOT backed up.
        budget = PlanningBudget(model_interval_cap=1, wall_budget_s=5.0)
        result = plan(state, task, context, H=4, budget=budget, capabilities=caps, cfg=self.cfg)

        self.assertFalse(result.completed)
        self.assertIsNotNone(result.selected_prefix)
        self.assertEqual(result.selected_prefix.raw_candidate_id, "cand_0")
        self.assertEqual(result.fallback_reason, "budget_exhausted")
        self.assertEqual(result.timing_counters["model_intervals"], 1)
        self.assertEqual(len(caps.predict_step_calls), 1)

        # H=0 terminates promptly with completed=True and fallback_reason="zero_horizon", independent of U
        for stop_val in (0.0, 0.5, 1.0):
            ticks_h0 = [0]
            def fake_clock_h0() -> float:
                ticks_h0[0] += 1
                if ticks_h0[0] > 10:
                    raise TimeoutError(f"Plan looped infinitely on H=0 (stop_val={stop_val})")
                return 0.0

            caps_h0 = DeterministicMockCapabilities(stop_value=stop_val)
            res_h0 = plan(
                state,
                task,
                context,
                H=0,
                budget=PlanningBudget(native_call_cap=5),
                capabilities=caps_h0,
                cfg=self.cfg,
                clock=fake_clock_h0,
            )
            self.assertTrue(res_h0.completed)
            self.assertIsNone(res_h0.selected_prefix)
            self.assertEqual(res_h0.fallback_reason, "zero_horizon")
            self.assertEqual(res_h0.call_count, 0)
            self.assertAlmostEqual(res_h0.expected_return, stop_val)

        # Absorbed root (H=4, stop_value=1.0) terminates promptly with completed=True and fallback_reason="absorbed_root"
        ticks_abs = [0]
        def fake_clock_abs() -> float:
            ticks_abs[0] += 1
            if ticks_abs[0] > 10:
                raise TimeoutError("Plan looped infinitely on absorbed root")
            return 0.0

        caps_absorbed = DeterministicMockCapabilities(stop_value=1.0)
        res_abs = plan(
            state,
            task,
            context,
            H=4,
            budget=PlanningBudget(native_call_cap=5),
            capabilities=caps_absorbed,
            cfg=self.cfg,
            clock=fake_clock_abs,
        )
        self.assertTrue(res_abs.completed)
        self.assertIsNone(res_abs.selected_prefix)
        self.assertEqual(res_abs.fallback_reason, "absorbed_root")
        self.assertEqual(res_abs.call_count, 0)
        self.assertAlmostEqual(res_abs.expected_return, 1.0)

    def test_audit_issue_3_native_observation_from_representative(self) -> None:
        """Representative observation passed to sample_prior is a validated native Observation with points, T_w_e, grip."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan
        from icgs.contracts.records import Observation

        caps = DeterministicMockCapabilities()
        state = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        plan(state, task, context, H=4, budget=PlanningBudget(native_call_cap=1), capabilities=caps, cfg=self.cfg)

        self.assertEqual(len(caps.sample_prior_calls), 1)
        obs = caps.sample_prior_calls[0]["obs"]
        self.assertIsInstance(obs, Observation)
        # Must be detached CPU owned numpy arrays, not torch tensors
        self.assertIsInstance(obs.points, np.ndarray)
        self.assertIsInstance(obs.T_w_e, np.ndarray)
        self.assertEqual(obs.points.ndim, 2)
        self.assertEqual(obs.points.shape[-1], 3)
        self.assertGreater(obs.points.shape[0], 0)
        self.assertTrue(np.all(np.isfinite(obs.points)))
        self.assertEqual(obs.T_w_e.shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(obs.T_w_e)))
        self.assertIsInstance(obs.grip, float)
        self.assertTrue(math.isfinite(obs.grip))
        self.assertIn(obs.grip, (0.0, 1.0))

    def test_audit_issue_4_config_discipline_and_no_duck_typing(self) -> None:
        """MethodConfig is strictly required for plan; formula helpers accept PlanningConfig."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan, widening_limit, uct

        with self.assertRaises(TypeError):
            widening_limit(0, object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            uct(1.0, 1, 1, object())  # type: ignore[arg-type]

        caps = DeterministicMockCapabilities()
        state = _make_physical_state()
        task = {"history_id": "task_0"}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(native_call_cap=1)

        with self.assertRaises(TypeError):
            plan(state, task, context, H=4, budget=budget, capabilities=caps, cfg="not_a_config")  # type: ignore[arg-type]

        # plan requires MethodConfig only; PlanningConfig alone must be rejected
        with self.assertRaises(TypeError):
            plan(state, task, context, H=4, budget=budget, capabilities=caps, cfg=self.cfg.planning)  # type: ignore[arg-type]

        # formula helpers accept PlanningConfig
        self.assertEqual(widening_limit(0, self.cfg.planning), 1)
        expected_uct = 1.0 + float(self.cfg.planning.uct_exploration) * math.sqrt(math.log(2))
        self.assertAlmostEqual(uct(1.0, 1, 1, self.cfg.planning), expected_uct)

    def test_audit_issue_5_strict_materialization_length_and_finite_completion(self) -> None:
        """Malformed completion or wrong prefix length from capabilities is strictly rejected without silent alteration."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan

        state = _make_physical_state()
        task = {"history_id": "task_0"}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(native_call_cap=1)

        # Invalid stop probability (> 1)
        caps_bad_stop = DeterministicMockCapabilities(stop_value=1.5)
        with self.assertRaises(ValueError):
            plan(state, task, context, H=4, budget=budget, capabilities=caps_bad_stop, cfg=self.cfg)

        # Multi-element stop probability at root
        caps_bad_shape_stop = DeterministicMockCapabilities(stop_value=np.array([0.2, 0.8]))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            plan(state, task, context, H=4, budget=budget, capabilities=caps_bad_shape_stop, cfg=self.cfg)

        # Leaf calibrated_value multi-element rejection
        class BadValShapeCapabilities(DeterministicMockCapabilities):
            def evaluate_state(self, state: PhysicalState, task: Any, events: Any, H: int) -> EvaluationOutput:
                return _make_evaluation_output(value=np.array([0.3, 0.7]), stop=self.stop_value, horizon=H)  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            plan(state, task, context, H=4, budget=budget, capabilities=BadValShapeCapabilities(), cfg=self.cfg)

        # Leaf calibrated_value > 1.0 rejection
        class BadValHighCapabilities(DeterministicMockCapabilities):
            def evaluate_state(self, state: PhysicalState, task: Any, events: Any, H: int) -> EvaluationOutput:
                return _make_evaluation_output(value=1.5, stop=self.stop_value, horizon=H)

        with self.assertRaises(ValueError):
            plan(state, task, context, H=4, budget=budget, capabilities=BadValHighCapabilities(), cfg=self.cfg)

        # Leaf calibrated_value < 0.0 rejection
        class BadValLowCapabilities(DeterministicMockCapabilities):
            def evaluate_state(self, state: PhysicalState, task: Any, events: Any, H: int) -> EvaluationOutput:
                return _make_evaluation_output(value=-0.5, stop=self.stop_value, horizon=H)

        with self.assertRaises(ValueError):
            plan(state, task, context, H=4, budget=budget, capabilities=BadValLowCapabilities(), cfg=self.cfg)

        # Invalid materialization length
        class BadPrefixCapabilities(DeterministicMockCapabilities):
            def materialize_prefix(self, candidate: Any, *, h: int, r: int, duration_s: float) -> CommandPrefix:
                # Return 1 command when h was requested
                cmds = (_make_timed_command(duration_s=duration_s),)
                return CommandPrefix(commands=cmds, proposal_root=np.eye(4), raw_candidate_id="bad_cand")

        caps_bad_prefix = BadPrefixCapabilities()
        with self.assertRaises(ValueError):
            plan(state, task, context, H=4, budget=budget, capabilities=caps_bad_prefix, cfg=self.cfg)

    def test_audit_issue_6_duplicate_audit_exposure_and_flags(self) -> None:
        """Preserve candidate ID from prefix; expose duplicate flag, ordering, and hash via recorder."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan

        cmd = (_make_timed_command(offset=0.1), _make_timed_command(offset=0.2))
        caps = DeterministicMockCapabilities(candidate_commands=[cmd, cmd])
        state = _make_physical_state()
        task = {"history_id": "task_0"}
        context = {"context_id": "ctx_0"}
        rec = TestRecorder()

        result = plan(
            state,
            task,
            context,
            H=4,
            budget=PlanningBudget(native_call_cap=2),
            capabilities=caps,
            cfg=self.cfg,
            recorder=rec,
        )

        self.assertEqual(result.audit_ids, ("cand_0", "cand_1"))
        audit_events = rec.find_events("candidate.audit")
        self.assertEqual(len(audit_events), 2)
        self.assertEqual(audit_events[0]["fields"]["candidate_id"], "cand_0")
        self.assertEqual(audit_events[0]["fields"]["order"], 0)
        self.assertFalse(audit_events[0]["fields"]["is_duplicate"])

        self.assertEqual(audit_events[1]["fields"]["candidate_id"], "cand_1")
        self.assertEqual(audit_events[1]["fields"]["order"], 1)
        self.assertTrue(audit_events[1]["fields"]["is_duplicate"])
        self.assertEqual(audit_events[0]["fields"]["commands_hash"], audit_events[1]["fields"]["commands_hash"])


class SearchBudgetAndBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = MethodConfig()

    def test_step_1_eligible_completion_deadline_equality(self) -> None:
        """Step 1 assertion body from task-3-brief.md."""
        from icgs.algorithms.planning.budget import eligible_completion

        self.assertTrue(eligible_completion(0.5, 0.5))
        self.assertFalse(eligible_completion(0.5001, 0.5))

    def test_deadline_equality_and_late_candidate_overshoot_rejection(self) -> None:
        """Operation completing at deadline equality is accepted; overshoot candidate is rejected."""
        import icgs.algorithms.planning.rerank as rerank_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        caps = MockCapabilities()
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        rec = TestRecorder()

        clock_ticks = [0.0]

        def clock_fn() -> float:
            return clock_ticks[0]

        # candidate 0 finishes at 0.5 (deadline 0.5), candidate 1 samples and finishes at 0.501
        orig_evaluate_state = caps.evaluate_state
        call_count = [0]

        def timed_evaluate_state(*args: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                # Root eval finished
                clock_ticks[0] = 0.1
            elif call_count[0] == 4:
                # Leaf eval for cand_0 finished: exactly at deadline
                clock_ticks[0] = 0.5
            return orig_evaluate_state(*args, **kwargs)

        orig_sample_prior = caps.sample_prior

        def timed_sample_prior(*args: Any, **kwargs: Any) -> Any:
            cand = orig_sample_prior(*args, **kwargs)
            if cand["index"] >= 1:
                # Candidate 1 samples after deadline: overshoot
                clock_ticks[0] = 0.501
            return cand

        caps.evaluate_state = timed_evaluate_state
        caps.sample_prior = timed_sample_prior

        budget = PlanningBudget(wall_budget_s=0.5)
        res_rerank = rerank_module.plan(
            state, task, context, H=2, budget=budget, capabilities=caps, cfg=self.cfg, clock=clock_fn, recorder=rec
        )
        self.assertTrue(res_rerank.completed)
        self.assertIsNotNone(res_rerank.selected_prefix)
        self.assertEqual(res_rerank.selected_prefix.raw_candidate_id, "cand_0")

        # Verify overshoot was recorded for candidate 1
        overshoot_events = rec.find_events("operation.overshoot")
        self.assertTrue(len(overshoot_events) > 0)
        self.assertEqual(overshoot_events[0]["fields"]["operation"], "sample_prior")

    def test_root_evaluation_overshoot_and_zero_horizon_timeout(self) -> None:
        """Root eval overshoot for H=0 yields timeout; for H>0 triggers fallback prefix."""
        import icgs.algorithms.planning.mcts as mcts_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        # H = 0 overshoots deadline at root evaluation
        caps_h0 = MockCapabilities()
        clock_h0 = [0.0]

        def timed_root_eval_h0(*args: Any, **kwargs: Any) -> Any:
            clock_h0[0] = 0.1001  # Exceeds 0.1 deadline
            return caps_h0._orig_eval(*args, **kwargs)

        caps_h0._orig_eval = caps_h0.evaluate_state
        caps_h0.evaluate_state = timed_root_eval_h0

        res_h0 = mcts_module.plan(
            state,
            task,
            context,
            H=0,
            budget=PlanningBudget(wall_budget_s=0.1),
            capabilities=caps_h0,
            cfg=self.cfg,
            clock=lambda: clock_h0[0],
        )
        self.assertFalse(res_h0.completed)
        self.assertEqual(res_h0.fallback_reason, "zero_horizon_timeout")
        self.assertIsNone(res_h0.selected_prefix)

        # H = 2 overshoots deadline at root evaluation: triggers fallback prefix
        caps_h2 = MockCapabilities()
        clock_h2 = [0.0]

        def timed_root_eval_h2(*args: Any, **kwargs: Any) -> Any:
            clock_h2[0] = 0.1001
            return caps_h2._orig_eval(*args, **kwargs)

        caps_h2._orig_eval = caps_h2.evaluate_state
        caps_h2.evaluate_state = timed_root_eval_h2

        res_h2 = mcts_module.plan(
            state,
            task,
            context,
            H=2,
            budget=PlanningBudget(wall_budget_s=0.1),
            capabilities=caps_h2,
            cfg=self.cfg,
            clock=lambda: clock_h2[0],
        )
        self.assertFalse(res_h2.completed)
        self.assertEqual(res_h2.fallback_reason, "budget_exhausted")
        self.assertIsNotNone(res_h2.selected_prefix)
        self.assertEqual(len(res_h2.selected_prefix.commands), 2)

    def test_predict_step_and_leaf_eval_overshoot(self) -> None:
        """Predict_step or leaf eval overshooting deadline interrupts and triggers fallback."""
        import icgs.algorithms.planning.mcts as mcts_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        caps = MockCapabilities()
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        rec = TestRecorder()

        clock_val = [0.0]

        def clock_fn() -> float:
            return clock_val[0]

        orig_predict = caps.predict_step

        def timed_predict(*args: Any, **kwargs: Any) -> Any:
            clock_val[0] = 0.51  # Exceeds 0.5 deadline
            return orig_predict(*args, **kwargs)

        caps.predict_step = timed_predict

        budget = PlanningBudget(wall_budget_s=0.5)
        res = mcts_module.plan(
            state, task, context, H=2, budget=budget, capabilities=caps, cfg=self.cfg, clock=clock_fn, recorder=rec
        )
        self.assertFalse(res.completed)
        self.assertEqual(res.fallback_reason, "budget_exhausted")
        # Candidate 0 was sampled before predict_step overshoot, so fallback reuses cand_0
        self.assertIsNotNone(res.selected_prefix)
        self.assertEqual(res.selected_prefix.raw_candidate_id, "cand_0")

        overshoots = rec.find_events("operation.overshoot")
        self.assertTrue(len(overshoots) > 0)
        self.assertEqual(overshoots[0]["fields"]["operation"], "predict_step")

    def test_zero_budget_caps_native_and_model_independent_counters(self) -> None:
        """Zero native cap draws exactly 1 fallback reference; zero model cap reuses sampled candidate."""
        import icgs.algorithms.planning.mcts as mcts_module
        import icgs.algorithms.planning.rerank as rerank_module
        import icgs.algorithms.planning.shooting as shooting_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        # Zero native call cap: search cannot sample any proposals. Fallback draws exactly 1 reference sample.
        for planner in (mcts_module.plan, rerank_module.plan, shooting_module.plan):
            caps = MockCapabilities()
            budget = PlanningBudget(native_call_cap=0)
            res = planner(state, task, context, H=2, budget=budget, capabilities=caps, cfg=self.cfg)
            self.assertFalse(res.completed)
            self.assertEqual(res.fallback_reason, "budget_exhausted")
            self.assertIsNotNone(res.selected_prefix)
            self.assertEqual(len(caps.sample_prior_calls), 1)
            self.assertEqual(res.timing_counters["native_calls"], 1)
            self.assertEqual(res.timing_counters["model_intervals"], 0)

        # Zero model interval cap with native_call_cap=1: search samples 1 proposal, but cannot step model. Fallback reuses cand_0 without new sample.
        for planner in (mcts_module.plan, rerank_module.plan, shooting_module.plan):
            caps = MockCapabilities()
            budget = PlanningBudget(native_call_cap=1, model_interval_cap=0)
            res = planner(state, task, context, H=2, budget=budget, capabilities=caps, cfg=self.cfg)
            self.assertFalse(res.completed)
            self.assertEqual(res.fallback_reason, "budget_exhausted")
            self.assertIsNotNone(res.selected_prefix)
            self.assertEqual(res.selected_prefix.raw_candidate_id, "cand_0")
            self.assertEqual(len(caps.sample_prior_calls), 1)
            self.assertEqual(res.timing_counters["native_calls"], 1)
            self.assertEqual(res.timing_counters["model_intervals"], 0)

    def test_invalid_root_state_aborts_without_geometry_fabrication(self) -> None:
        """Invalid physical state or NaN points aborts immediately without fabricating geometry."""
        import icgs.algorithms.planning.mcts as mcts_module
        import icgs.algorithms.planning.rerank as rerank_module
        import icgs.algorithms.planning.shooting as shooting_module
        from icgs.algorithms.planning.budget import PlanningBudget

        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        caps = MockCapabilities()
        budget = PlanningBudget(native_call_cap=1)

        # Non-PhysicalState
        for planner in (mcts_module.plan, rerank_module.plan, shooting_module.plan):
            with self.assertRaises(TypeError):
                planner("invalid_state", task, context, H=2, budget=budget, capabilities=caps, cfg=self.cfg)

        # PhysicalState with NaN points
        bad_state = _make_physical_state(boundary=0)
        bad_state.x[0, 0, 0] = float("nan")
        for planner in (mcts_module.plan, rerank_module.plan, shooting_module.plan):
            with self.assertRaises(ValueError):
                planner(bad_state, task, context, H=2, budget=budget, capabilities=caps, cfg=self.cfg)

    def test_nonfinite_model_error_triggers_fallback_not_crash(self) -> None:
        """Nonfinite model outputs raise NonfiniteModelError and trigger fallback gracefully."""
        from icgs.algorithms.planning.budget import NonfiniteModelError, PlanningBudget
        import icgs.algorithms.planning.mcts as mcts_module
        import icgs.algorithms.planning.rerank as rerank_module
        import icgs.algorithms.planning.shooting as shooting_module

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        # Capabilities whose predict_terminal raises NonfiniteModelError
        for planner in (mcts_module.plan, rerank_module.plan, shooting_module.plan):
            caps = MockCapabilities()

            def failing_predict_terminal(*args: Any, **kwargs: Any) -> Any:
                raise NonfiniteModelError("Nonfinite terminal probabilities produced by model")

            caps.predict_terminal = failing_predict_terminal

            rec = TestRecorder()
            budget = PlanningBudget(native_call_cap=2)
            res = planner(state, task, context, H=2, budget=budget, capabilities=caps, cfg=self.cfg, recorder=rec)
            self.assertFalse(res.completed)
            self.assertEqual(res.fallback_reason, "nonfinite_model_error")
            self.assertIsNotNone(res.selected_prefix)
            model_errors = rec.find_events("model.error")
            self.assertTrue(len(model_errors) > 0)
            self.assertEqual(model_errors[0]["fields"]["error"], "nonfinite_model_output")

    def test_rerank_configured_h_and_native_T8(self) -> None:
        """Rerank supports configured h and full native T8 rollouts with tie breaking."""
        import icgs.algorithms.planning.rerank as rerank_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        # Default h = 2
        caps_h2 = MockCapabilities()
        budget = PlanningBudget(native_call_cap=2)
        res_h2 = rerank_module.plan(state, task, context, H=4, budget=budget, capabilities=caps_h2, cfg=self.cfg)
        self.assertTrue(res_h2.completed)
        self.assertIsNotNone(res_h2.selected_prefix)
        self.assertEqual(len(res_h2.selected_prefix.commands), 2)
        self.assertEqual(res_h2.timing_counters["iterations"], 2)

        # Native T8 = 8 steps rollout
        caps_t8 = MockCapabilities()
        cfg_t8 = MethodConfig(planning={"H": 8, "h": 2, "L": 8})
        res_t8 = rerank_module.plan(
            state, task, context, H=8, budget=budget, capabilities=caps_t8, cfg=cfg_t8, horizon=8
        )
        self.assertTrue(res_t8.completed)
        self.assertIsNotNone(res_t8.selected_prefix)
        self.assertEqual(len(res_t8.selected_prefix.commands), 8)

        # Tie breaking: if multiple candidates yield equal G, earliest candidate order is selected
        caps_ties = MockCapabilities()
        rec = TestRecorder()
        res_ties = rerank_module.plan(
            state, task, context, H=4, budget=budget, capabilities=caps_ties, cfg=self.cfg, recorder=rec
        )
        self.assertEqual(res_ties.selected_prefix.raw_candidate_id, "cand_0")

    def test_shooting_resampling_through_L_and_ties(self) -> None:
        """Shooting resamples prior at predicted nodes up to horizon L and breaks ties to earliest sequence."""
        import icgs.algorithms.planning.shooting as shooting_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        caps = MockCapabilities()

        # Sequence with L = 4 and h = 2 requires 2 chunks (2 sample_prior calls per sequence)
        budget = PlanningBudget(native_call_cap=4)
        res = shooting_module.plan(state, task, context, H=4, budget=budget, capabilities=caps, cfg=self.cfg)
        self.assertTrue(res.completed)
        self.assertIsNotNone(res.selected_prefix)
        # Result prefix is the root action prefix (length h = 2)
        self.assertEqual(len(res.selected_prefix.commands), 2)
        self.assertEqual(res.selected_prefix.raw_candidate_id, "cand_0")
        # 4 native calls = 2 completed sequences of 2 chunks each
        self.assertEqual(len(caps.sample_prior_calls), 4)
        self.assertEqual(res.timing_counters["iterations"], 2)

        # Verify that sample 1 received the predicted node observation (boundary 2)
        obs_1 = caps.sample_prior_calls[1]["obs"]
        self.assertTrue(np.all(np.isfinite(obs_1.points)))
        self.assertTrue(np.all(np.isfinite(obs_1.T_w_e)))

    def test_matched_counters_and_exact_cache_across_all_planners(self) -> None:
        """MCTS, rerank, and shooting produce matched counters and respect exact cache hits."""
        import icgs.algorithms.planning.mcts as mcts_module
        import icgs.algorithms.planning.rerank as rerank_module
        import icgs.algorithms.planning.shooting as shooting_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(native_call_cap=2)

        for planner_name, planner_fn in (
            ("mcts", mcts_module.plan),
            ("rerank", rerank_module.plan),
            ("shooting", shooting_module.plan),
        ):
            caps = MockCapabilities()
            res = planner_fn(state, task, context, H=4, budget=budget, capabilities=caps, cfg=self.cfg)
            self.assertTrue(res.completed, f"{planner_name} should be completed")
            # Verify counter keys match across all 3
            self.assertEqual(
                set(res.timing_counters.keys()),
                {
                    "iterations",
                    "native_calls",
                    "model_intervals",
                    "evaluator_calls",
                    "task_tracker_calls",
                    "terminal_calls",
                    "materialization_calls",
                    "sync_calls",
                },
                f"{planner_name} timing_counters keys mismatch",
            )
            for k, v in res.timing_counters.items():
                self.assertIsInstance(v, int, f"{planner_name} {k} must be int")
                self.assertGreaterEqual(v, 0, f"{planner_name} {k} must be >= 0")

            # Verify device synchronization was called
            self.assertGreater(caps.sync_calls, 0, f"{planner_name} must invoke device synchronization")


class SearchTask3FixRound1Tests(unittest.TestCase):
    """Regressions for Task 3 Fix Round 1 audit findings."""

    def setUp(self) -> None:
        self.cfg = MethodConfig()

    def test_audit_issue_1_late_absorbed_root_for_h_positive_triggers_fallback(self) -> None:
        """For H > 0, late root eval with stop=1 must not suppress reference fallback."""
        import icgs.algorithms.planning.mcts as mcts_module
        import icgs.algorithms.planning.rerank as rerank_module
        import icgs.algorithms.planning.shooting as shooting_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(wall_budget_s=0.5)

        for planner_name, planner_fn in (
            ("mcts", mcts_module.plan),
            ("rerank", rerank_module.plan),
            ("shooting", shooting_module.plan),
        ):
            caps = MockCapabilities(stop_value=1.0)
            times_copy = [0.0, 0.0, 0.6, 0.6, 0.6, 0.6]
            res = planner_fn(
                state,
                task,
                context,
                H=4,
                budget=budget,
                capabilities=caps,
                cfg=self.cfg,
                clock=lambda: times_copy.pop(0) if times_copy else 0.6,
            )
            self.assertFalse(res.completed, f"{planner_name} should not be marked completed")
            self.assertIsNotNone(res.selected_prefix, f"{planner_name} must materialize fallback prefix")
            self.assertEqual(len(res.selected_prefix.commands), self.cfg.planning.h)
            self.assertEqual(res.fallback_reason, "budget_exhausted")

    def test_audit_issue_2_track_task_late_prevents_terminal_start_and_materialization_timing(self) -> None:
        """Overrunning track_task must prevent predict_terminal from starting."""
        import icgs.algorithms.planning.mcts as mcts_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        caps = MockCapabilities()
        budget = PlanningBudget(wall_budget_s=0.5)
        recorder = TestRecorder()

        current_time = [0.0]
        def clock():
            return current_time[0]

        original_track = caps.track_task
        def track_task_with_delay(prev, st, ev):
            current_time[0] = 0.6  # Past 0.5 deadline
            return original_track(prev, st, ev)
        caps.track_task = track_task_with_delay

        res = mcts_module.plan(
            state,
            task,
            context,
            H=4,
            budget=budget,
            capabilities=caps,
            cfg=self.cfg,
            clock=clock,
            recorder=recorder,
        )
        self.assertFalse(res.completed)
        self.assertEqual(len(caps.predict_terminal_calls), 0, "predict_terminal must not be called after track_task overshoots")

    def test_audit_issue_3_validate_context_called_before_root_eval_and_invalid_state_fields(self) -> None:
        """Root geometry and context validated before root eval; nonfinite predicted state fields caught."""
        import icgs.algorithms.planning.mcts as mcts_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        budget = PlanningBudget(wall_budget_s=0.5)

        class ContextValidatingCaps(MockCapabilities):
            def validate_context(self, ctx):
                if ctx.get("context_id") != "valid_ctx":
                    raise ValueError("invalid context provided")

        caps = ContextValidatingCaps()
        # Even with H=0, invalid context must be rejected!
        with self.assertRaisesRegex(ValueError, "invalid context"):
            mcts_module.plan(state, task, {"context_id": "bad_ctx"}, H=0, budget=budget, capabilities=caps, cfg=self.cfg)

    def test_audit_issue_4_rerank_horizon_strict_types_and_budget_audit(self) -> None:
        """Rerank strictly enforces int type and (h, P) values; planning.start audits resolved config."""
        import icgs.algorithms.planning.rerank as rerank_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(wall_budget_s=0.5)
        caps = MockCapabilities()
        rec = TestRecorder()

        # Non-int (float/bool) rejected
        with self.assertRaises(TypeError):
            rerank_module.plan(state, task, context, H=4, budget=budget, capabilities=caps, cfg=self.cfg, horizon=3.5)

        # Non-allowed horizon value rejected (neither h=2 nor P=8)
        with self.assertRaises(ValueError):
            rerank_module.plan(state, task, context, H=4, budget=budget, capabilities=caps, cfg=self.cfg, horizon=5)

        # Audit planning.start has config_sha256 and effective budgets
        rerank_module.plan(state, task, context, H=4, budget=budget, capabilities=caps, cfg=self.cfg, horizon=2, recorder=rec)
        start_events = [e for e in rec.events if e.get("name") == "planning.start"]
        self.assertTrue(len(start_events) > 0)
        fields = start_events[0]["fields"]
        self.assertEqual(fields["config_sha256"], self.cfg.fingerprint())
        self.assertIn("wall_budget_s", fields)
        self.assertIn("clock_track", fields)

    def test_audit_issue_5_leaf_head_equality_vs_late_and_shooting_node_provenance(self) -> None:
        """Leaf evaluation at deadline equality accepted; leaf evaluation overshooting deadline rejected."""
        import icgs.algorithms.planning.rerank as rerank_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(wall_budget_s=0.5)

        # Rerank with H=4, h=2 (so rollout reaches leaf with remaining=2 > 0, evaluating heads)
        # Test A: Leaf head 2 finishes at exact 0.5 -> accepted!
        caps_a = MockCapabilities()
        clock_ticks_a = [0.0]
        eval_count_a = [0]
        orig_eval_a = caps_a.evaluate_state

        def timed_eval_a(*args: Any, **kwargs: Any) -> Any:
            eval_count_a[0] += 1
            if eval_count_a[0] == 1:
                clock_ticks_a[0] = 0.05
            elif eval_count_a[0] == 4:
                clock_ticks_a[0] = 0.5
            return orig_eval_a(*args, **kwargs)

        caps_a.evaluate_state = timed_eval_a
        res_a = rerank_module.plan(
            state,
            task,
            context,
            H=4,
            budget=budget,
            capabilities=caps_a,
            cfg=self.cfg,
            horizon=2,
            clock=lambda: clock_ticks_a[0],
        )
        self.assertTrue(res_a.completed, "Rollout finishing at exact deadline equality must be completed")

        # Test B: Leaf head 2 finishes at 0.5001 -> overshoots!
        caps_b = MockCapabilities()
        clock_ticks_b = [0.0]
        eval_count_b = [0]
        orig_eval_b = caps_b.evaluate_state

        def timed_eval_b(*args: Any, **kwargs: Any) -> Any:
            eval_count_b[0] += 1
            if eval_count_b[0] == 1:
                clock_ticks_b[0] = 0.05
            elif eval_count_b[0] == 4:
                clock_ticks_b[0] = 0.5001
            return orig_eval_b(*args, **kwargs)

        caps_b.evaluate_state = timed_eval_b
        res_b = rerank_module.plan(
            state,
            task,
            context,
            H=4,
            budget=budget,
            capabilities=caps_b,
            cfg=self.cfg,
            horizon=2,
            clock=lambda: clock_ticks_b[0],
        )
        self.assertFalse(res_b.completed, "Rollout overshooting deadline must be rejected")

    def test_audit_issue_5_no_late_mcts_backup_and_shooting_node_provenance(self) -> None:
        """No late MCTS backup if iteration overshoots; shooting passes predicted node pose and breaks ties."""
        import icgs.algorithms.planning.mcts as mcts_module
        import icgs.algorithms.planning.shooting as shooting_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(wall_budget_s=0.5)

        # 1. MCTS no late backup
        caps_mcts = MockCapabilities()
        clock_ticks_m = [0.0]
        eval_count_m = [0]
        orig_eval_m = caps_mcts.evaluate_state

        def timed_eval_m(*args: Any, **kwargs: Any) -> Any:
            eval_count_m[0] += 1
            if eval_count_m[0] == 1:
                clock_ticks_m[0] = 0.05
            elif eval_count_m[0] == 4:
                clock_ticks_m[0] = 0.6
            return orig_eval_m(*args, **kwargs)

        caps_mcts.evaluate_state = timed_eval_m
        rec_mcts = TestRecorder()
        res_mcts = mcts_module.plan(
            state,
            task,
            context,
            H=4,
            budget=budget,
            capabilities=caps_mcts,
            cfg=self.cfg,
            clock=lambda: clock_ticks_m[0],
            recorder=rec_mcts,
        )
        self.assertFalse(res_mcts.completed)
        self.assertEqual(res_mcts.timing_counters["iterations"], 0)

        # 2. Shooting passes predicted node pose to sample_prior at intermediate node
        caps_shoot = MockCapabilities()
        res_shoot = shooting_module.plan(
            state,
            task,
            context,
            H=4,
            budget=PlanningBudget(native_call_cap=2),
            capabilities=caps_shoot,
            cfg=self.cfg,
        )
        self.assertTrue(res_shoot.completed)
        self.assertEqual(len(caps_shoot.sample_prior_calls), 2)
        obs_1 = caps_shoot.sample_prior_calls[1]["obs"]
        self.assertFalse(np.allclose(obs_1.T_w_e, state.T_w_e[0]))

    def test_audit_issue_6_returned_invalid_state_and_fallback_proposal_root(self) -> None:
        """Returned invalid state fields trigger NonfiniteModelError; fallback has root pose."""
        import icgs.algorithms.planning.mcts as mcts_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(wall_budget_s=0.5)

        class NaNStateCaps(MockCapabilities):
            def predict_step(self, st, cmd, *, head_id):
                st_corrupt = _make_physical_state(boundary=st.boundary + 1)
                st_corrupt.x[0, 0, 0] = float("nan")
                return _make_prediction(st_corrupt, head_id)

        caps = NaNStateCaps()
        rec = TestRecorder()
        res = mcts_module.plan(
            state,
            task,
            context,
            H=4,
            budget=budget,
            capabilities=caps,
            cfg=self.cfg,
            recorder=rec,
        )
        self.assertFalse(res.completed)
        model_errors = [e for e in rec.events if e.get("name") == "model.error"]
        self.assertTrue(len(model_errors) > 0)
        self.assertIsNotNone(res.selected_prefix)
        self.assertTrue(np.allclose(res.selected_prefix.proposal_root, state.T_w_e[0]))


class SearchTask3FixRound2Tests(unittest.TestCase):
    """Regressions for Task 3 Fix Round 2 audit findings."""

    def setUp(self) -> None:
        from icgs.configuration.method import MethodConfig
        from pathlib import Path
        primary_json = Path(__file__).resolve().parent.parent / "src/icgs/configuration/profiles/icgs_primary.json"
        self.cfg = MethodConfig.from_file(primary_json)

    def test_regression_1_completed_candidate_preserved_after_subsequent_model_error(self) -> None:
        """Completed results are preserved and selected even if a subsequent model error occurs across ALL 3 planners."""
        import icgs.algorithms.planning.mcts as mcts_module
        import icgs.algorithms.planning.rerank as rerank_module
        import icgs.algorithms.planning.shooting as shooting_module
        from icgs.algorithms.planning.budget import PlanningBudget, NonfiniteModelError

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(native_call_cap=3)

        # 1. Rerank test: cand_0 succeeds, cand_1 throws NonfiniteModelError
        class ErrorOnCand1Caps(DeterministicMockCapabilities):
            def __init__(self):
                super().__init__()
                self.eval_call_count = 0

            def evaluate_state(self, st, t, ev, H):
                self.eval_call_count += 1
                if self.eval_call_count > 4:  # root + leaf for cand_0 succeed, then fail on cand_1
                    raise NonfiniteModelError("leaf head nonfinite")
                return _make_evaluation_output(value=0.8, stop=0.1, horizon=H)

        caps = ErrorOnCand1Caps()
        rec_rerank = TestRecorder()
        res_rerank = rerank_module.plan(
            state, task, context, H=4, budget=budget, capabilities=caps, cfg=self.cfg, recorder=rec_rerank
        )
        self.assertTrue(res_rerank.completed, "Rerank must complete with candidate 0")
        self.assertIsNone(res_rerank.fallback_reason)
        self.assertEqual(res_rerank.selected_prefix.raw_candidate_id, "cand_0")

        # 2. MCTS test: iter 1 completes cand_0, iter 2 encounters NonfiniteModelError
        class ErrorOnIter2MCTSCaps(DeterministicMockCapabilities):
            def __init__(self):
                super().__init__()
                self.eval_call_count = 0

            def evaluate_state(self, st, t, ev, H):
                self.eval_call_count += 1
                if self.eval_call_count > 4:  # root + leaf for iter 1 succeed
                    raise NonfiniteModelError("mcts leaf nonfinite")
                return _make_evaluation_output(value=0.8, stop=0.1, horizon=H)

        caps_mcts = ErrorOnIter2MCTSCaps()
        rec_mcts = TestRecorder()
        res_mcts = mcts_module.plan(
            state, task, context, H=4, budget=budget, capabilities=caps_mcts, cfg=self.cfg, recorder=rec_mcts
        )
        self.assertTrue(res_mcts.completed, "MCTS must complete with edge 0")
        self.assertIsNone(res_mcts.fallback_reason)
        self.assertEqual(res_mcts.selected_prefix.raw_candidate_id, "cand_0")

        # 3. Shooting test: seq 0 completes, seq 1 encounters NonfiniteModelError on predict_step
        class ErrorOnSeq2ShootingCaps(DeterministicMockCapabilities):
            def __init__(self):
                super().__init__()
                self.step_call_count = 0

            def predict_step(self, st, cmd, *, head_id):
                self.step_call_count += 1
                # First sequence completes: 2 chunks * 2 commands * 3 heads = 12 predict_step calls.
                # In second sequence, predict_step raises NonfiniteModelError.
                if self.step_call_count > 12:
                    raise NonfiniteModelError("shooting predict_step nonfinite")
                return super().predict_step(st, cmd, head_id=head_id)

        caps_shoot = ErrorOnSeq2ShootingCaps()
        rec_shoot = TestRecorder()
        res_shoot = shooting_module.plan(
            state, task, context, H=4, budget=budget, capabilities=caps_shoot, cfg=self.cfg, recorder=rec_shoot
        )
        self.assertTrue(
            any(e.get("name") == "model.error" for e in rec_shoot.events),
            "Shooting must record model.error on subsequent error",
        )
        self.assertTrue(res_shoot.completed, "Shooting must complete with seq 0")
        self.assertIsNone(res_shoot.fallback_reason)
        self.assertEqual(res_shoot.selected_prefix.raw_candidate_id, "cand_0")

    def test_regression_2_late_sampled_prior_reused_in_fallback(self) -> None:
        """A sampled prior that completes after search deadline is preserved as earliest_candidate for fallback across ALL 3 planners."""
        import icgs.algorithms.planning.mcts as mcts_module
        import icgs.algorithms.planning.rerank as rerank_module
        import icgs.algorithms.planning.shooting as shooting_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(wall_budget_s=0.5)

        for planner_name, planner_fn in [
            ("rerank", rerank_module.plan),
            ("mcts", mcts_module.plan),
            ("shooting", shooting_module.plan),
        ]:
            t_val = [0.0]
            def fake_clock():
                return t_val[0]

            caps = DeterministicMockCapabilities()
            orig_sample = caps.sample_prior
            def late_sample(*args, **kwargs):
                t_val[0] = 0.501  # overshoots search deadline
                return orig_sample(*args, **kwargs)

            caps.sample_prior = late_sample

            rec = TestRecorder()
            res = planner_fn(
                state, task, context, H=4, budget=budget, capabilities=caps, cfg=self.cfg, clock=fake_clock, recorder=rec
            )
            self.assertFalse(res.completed, f"{planner_name} must not be completed on late sample")
            self.assertEqual(res.fallback_reason, "budget_exhausted")
            # Exactly ONE sample_prior call should have occurred (the late one was reused in fallback)
            self.assertEqual(
                len(caps.sample_prior_calls),
                1,
                f"{planner_name} must reuse late sampled prior in fallback without drawing a second sample",
            )

    def test_regression_3_fallback_rejects_wrong_root_without_mutation(self) -> None:
        """Fallback rejects prefix with mismatching proposal_root without mutating frozen object."""
        from icgs.algorithms.planning.budget import execute_fallback, PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(wall_budget_s=0.5)

        bad_prefix_holder = []

        # Caps materializer returns proposal_root offset from root pose
        class WrongRootCaps(DeterministicMockCapabilities):
            def materialize_prefix(self, candidate, *, h, r, duration_s):
                bad_root = np.eye(4, dtype=np.float64)
                bad_root[0, 3] = 999.0  # wrong root
                cmds = tuple(_make_timed_command(duration_s=duration_s) for _ in range(h))
                p = CommandPrefix(commands=cmds, proposal_root=bad_root, raw_candidate_id="bad_cand")
                bad_prefix_holder.append(p)
                return p

        caps = WrongRootCaps()
        rec = TestRecorder()
        with self.assertRaises(ValueError):
            execute_fallback(
                state,
                task,
                context,
                H=4,
                budget=budget,
                capabilities=caps,
                cfg=self.cfg,
                seed=42,
                clock_fn=lambda: 0.1,
                start_time=0.0,
                deadline=0.5,
                earliest_candidate=None,
                timing_counters={},
                cache_counters={},
                audit_ids=(),
                recorder=rec,
            )
        self.assertEqual(len(bad_prefix_holder), 1)
        # Verify proposal_root was NOT mutated to match root pose
        self.assertEqual(bad_prefix_holder[0].proposal_root[0, 3], 999.0)

    def test_residual_4_panel_status_not_applied_and_envelope(self) -> None:
        """Panel dimensions record not_applied, matched, override, and per-planner algorithm_params."""
        import icgs.algorithms.planning.mcts as mcts_module
        import icgs.algorithms.planning.rerank as rerank_module
        import icgs.algorithms.planning.shooting as shooting_module
        from icgs.algorithms.planning.budget import PlanningBudget

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        # 1. Rerank with matched wall budget and not_applied native/model
        budget_matched = PlanningBudget(wall_budget_s=0.5)
        caps = DeterministicMockCapabilities()
        rec_rerank = TestRecorder()
        res_rerank = rerank_module.plan(
            state, task, context, H=4, budget=budget_matched, capabilities=caps, cfg=self.cfg, recorder=rec_rerank
        )
        self.assertTrue(res_rerank.completed)
        ev_rerank = [e for e in rec_rerank.events if e.get("name") == "planning.start"][0]["fields"]
        self.assertIn("resolved_config", ev_rerank)
        self.assertEqual(ev_rerank["wall_panel_status"], "matched")
        self.assertEqual(ev_rerank["native_panel_status"], "not_applied")
        self.assertEqual(ev_rerank["model_panel_status"], "not_applied")
        self.assertEqual(ev_rerank["clock_track_status"], "matched")
        self.assertTrue(ev_rerank["panel_matched"])
        self.assertEqual(ev_rerank["H"], 4)
        self.assertEqual(ev_rerank["boundary"], 0)
        self.assertIn("rollout_bound", ev_rerank["algorithm_params"])
        self.assertIn("h", ev_rerank["algorithm_params"])

        # 2. MCTS with wall budget override (e.g. 999.0s not in wall_budgets_s)
        budget_override = PlanningBudget(wall_budget_s=999.0, native_call_cap=2)
        rec_mcts = TestRecorder()
        res_mcts = mcts_module.plan(
            state, task, context, H=4, budget=budget_override, capabilities=caps, cfg=self.cfg, recorder=rec_mcts
        )
        self.assertTrue(res_mcts.completed)
        ev_mcts = [e for e in rec_mcts.events if e.get("name") == "planning.start"][0]["fields"]
        self.assertEqual(ev_mcts["wall_panel_status"], "override")
        self.assertFalse(ev_mcts["panel_matched"], "panel_matched must be False on override")
        self.assertIn("uct_exploration", ev_mcts["algorithm_params"])
        self.assertIn("h", ev_mcts["algorithm_params"])

        # 3. Shooting audit algorithm_params
        rec_shoot = TestRecorder()
        res_shoot = shooting_module.plan(
            state, task, context, H=4, budget=budget_matched, capabilities=caps, cfg=self.cfg, recorder=rec_shoot
        )
        self.assertTrue(res_shoot.completed)
        ev_shoot = [e for e in rec_shoot.events if e.get("name") == "planning.start"][0]["fields"]
        self.assertIn("L", ev_shoot["algorithm_params"])
        self.assertIn("h", ev_shoot["algorithm_params"])

    def test_residual_3_corrupted_physical_state_mutation_triggers_fallback(self) -> None:
        """Nonmutating validation preserves root tensor data_ptr, filters padding, and catches field-by-field corruption."""
        import icgs.algorithms.planning.mcts as mcts_module
        from icgs.algorithms.planning.budget import (
            PlanningBudget,
            validate_root_and_context,
            validate_predicted_state,
            NonfiniteModelError,
        )

        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}
        budget = PlanningBudget(wall_budget_s=0.5)
        caps = DeterministicMockCapabilities()

        # 1. Live state preservation invariant: validation must NOT change tensor data_ptr
        x_ptr_before = state.x.data_ptr()
        X_ptr_before = state.X.data_ptr()
        pose_ptr_before = state.T_w_e.data_ptr()
        valid_ptr_before = state.valid.data_ptr()

        obs = validate_root_and_context(state, context, caps)

        self.assertEqual(state.x.data_ptr(), x_ptr_before, "validate_root_and_context must not reallocate state.x")
        self.assertEqual(state.X.data_ptr(), X_ptr_before, "validate_root_and_context must not reallocate state.X")
        self.assertEqual(state.T_w_e.data_ptr(), pose_ptr_before, "validate_root_and_context must not reallocate state.T_w_e")
        self.assertEqual(state.valid.data_ptr(), valid_ptr_before, "validate_root_and_context must not reallocate state.valid")

        # 2. Masked padding absent from Observation: valid points filtered by mask
        cloud = np.random.randn(20, 3).astype(np.float32)
        cloud_mask = np.zeros(20, dtype=bool)
        cloud_mask[:7] = True
        state_masked = _make_physical_state(boundary=0, points=cloud, valid_mask=cloud_mask)
        obs_masked = validate_root_and_context(state_masked, context, caps)
        self.assertEqual(obs_masked.points.shape, (7, 3), "Observation points must filter out invalid padding from cached cloud")

        state_x_only = _make_physical_state(boundary=0)
        object.__setattr__(state_x_only, "cached_world_cloud", None)
        object.__setattr__(state_x_only, "cached_world_cloud_valid", None)
        state_x_only.valid[0, :10] = True
        state_x_only.valid[0, 10:] = False
        obs_x_only = validate_root_and_context(state_x_only, context, caps)
        self.assertEqual(obs_x_only.points.shape, (10, 3), "Observation points must filter out invalid padding from state.x")

        # 3. Field-by-field corrupted predicted states raise NonfiniteModelError
        valid_pred_state = _make_physical_state(boundary=1)
        validate_predicted_state(valid_pred_state)  # valid must pass

        # Test X corruption
        corrupt_X = _make_physical_state(boundary=1)
        corrupt_X.X[0, 0, 0] = float("nan")
        with self.assertRaises(NonfiniteModelError):
            validate_predicted_state(corrupt_X)

        # Test x corruption
        corrupt_x = _make_physical_state(boundary=1)
        corrupt_x.x[0, 0, 0] = float("nan")
        with self.assertRaises(NonfiniteModelError):
            validate_predicted_state(corrupt_x)

        # Test p corruption
        corrupt_p = _make_physical_state(boundary=1)
        corrupt_p.p[0, 0] = float("nan")
        with self.assertRaises(NonfiniteModelError):
            validate_predicted_state(corrupt_p)

        # Test memory corruption
        corrupt_mem = _make_physical_state(boundary=1)
        corrupt_mem.memory[0, 0, 0] = float("nan")
        with self.assertRaises(NonfiniteModelError):
            validate_predicted_state(corrupt_mem)

        # Test cached_world_cloud corruption
        corrupt_cloud = _make_physical_state(boundary=1)
        object.__setattr__(corrupt_cloud, "cached_world_cloud", torch.zeros((1, 10, 3), dtype=torch.float32))
        object.__setattr__(corrupt_cloud, "cached_world_cloud_valid", torch.ones((1, 10), dtype=torch.bool))
        corrupt_cloud.cached_world_cloud[0, 0, 0] = float("nan")
        with self.assertRaises(NonfiniteModelError):
            validate_predicted_state(corrupt_cloud)

        # Test cached_world_cloud_valid corruption
        corrupt_cloud_mask = _make_physical_state(boundary=1)
        object.__setattr__(corrupt_cloud_mask, "cached_world_cloud", torch.zeros((1, 10, 3), dtype=torch.float32))
        object.__setattr__(corrupt_cloud_mask, "cached_world_cloud_valid", torch.zeros((1, 10), dtype=torch.float32))  # non-bool
        with self.assertRaises(NonfiniteModelError):
            validate_predicted_state(corrupt_cloud_mask)

        # Test valid mask corruption (all False)
        corrupt_mask = _make_physical_state(boundary=1)
        corrupt_mask.valid[:] = False
        with self.assertRaises(NonfiniteModelError):
            validate_predicted_state(corrupt_mask)

        # Test T_w_e corruption (non-orthonormal)
        corrupt_pose = _make_physical_state(boundary=1)
        corrupt_pose.T_w_e[0, 0, 0] = 5.0
        with self.assertRaises(NonfiniteModelError):
            validate_predicted_state(corrupt_pose)

        # Test grip corruption (value 2.0)
        corrupt_grip = _make_physical_state(boundary=1)
        corrupt_grip.grip[0, 0] = 2.0
        with self.assertRaises(NonfiniteModelError):
            validate_predicted_state(corrupt_grip)

        # End-to-end planner fallback on NaN root eval
        class NaNRootEvalCaps(DeterministicMockCapabilities):
            def evaluate_state(self, st, t, ev, H):
                out = _make_evaluation_output(value=0.5, stop=0.1, horizon=H)
                object.__setattr__(out, "calibrated_value", np.array([float("nan")]))
                return out

        res_root = mcts_module.plan(
            state, task, context, H=4, budget=budget, capabilities=NaNRootEvalCaps(), cfg=self.cfg
        )
        self.assertFalse(res_root.completed)
        self.assertEqual(res_root.fallback_reason, "nonfinite_model_error")
        self.assertIsNotNone(res_root.selected_prefix)
        self.assertTrue(np.allclose(res_root.selected_prefix.proposal_root, state.T_w_e[0]))

    def test_residual_2_timed_operation_start_audit_and_attempt_counters(self) -> None:
        """timed_operation and fallback handle attempt/sync counters, start/error audits, and op/sync double failure."""
        import torch
        from icgs.algorithms.planning.budget import (
            timed_operation,
            execute_fallback,
            PlanningBudget,
            NonfiniteModelError,
        )

        budget = PlanningBudget(wall_budget_s=0.5)
        rec = TestRecorder()
        counters = {"native_calls": 0, "materialization_calls": 0, "sync_calls": 0}

        # 1. Operation starts before deadline: emits operation.started and increments counter
        caps = DeterministicMockCapabilities()
        res, finished, eligible = timed_operation(
            "sample_prior",
            lambda: "result_1",
            clock_fn=lambda: 0.1,
            deadline=0.5,
            budget=budget,
            capabilities=caps,
            recorder=rec,
            timing_counters=counters,
            counter_key="native_calls",
        )
        self.assertTrue(eligible)
        self.assertEqual(res, "result_1")
        self.assertEqual(counters["native_calls"], 1)
        self.assertEqual(counters["sync_calls"], 1)
        started_events = [e for e in rec.events if e.get("name") == "operation.started"]
        self.assertEqual(len(started_events), 1)
        self.assertEqual(started_events[0]["fields"]["operation"], "sample_prior")

        # 2. Clock already at or past deadline: op does NOT run, no started event emitted, counter NOT incremented
        res_late, finished_late, eligible_late = timed_operation(
            "sample_prior",
            lambda: "result_2",
            clock_fn=lambda: 0.5001,
            deadline=0.5,
            budget=budget,
            capabilities=caps,
            recorder=rec,
            timing_counters=counters,
            counter_key="native_calls",
        )
        self.assertFalse(eligible_late)
        self.assertIsNone(res_late)
        self.assertEqual(counters["native_calls"], 1)  # not attempted, so still 1

        # 3. Fallback permits beyond deadline: runs, emits operation.started and completed, increments counter
        res_perm, fin_perm, elig_perm = timed_operation(
            "sample_prior",
            lambda: "fallback_result",
            clock_fn=lambda: 0.6,
            deadline=0.5,
            budget=budget,
            capabilities=caps,
            recorder=rec,
            timing_counters=counters,
            counter_key="native_calls",
            permit_beyond_deadline=True,
        )
        self.assertTrue(elig_perm)
        self.assertEqual(res_perm, "fallback_result")
        self.assertEqual(counters["native_calls"], 2)

        # 4. Fallback execution failure: sample_prior throws; counters increment, error emitted
        state = _make_physical_state(boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        class FailingSampleCaps(DeterministicMockCapabilities):
            def sample_prior(self, *args, **kwargs):
                raise RuntimeError("prior sampling failed")

        fb_counters = {"native_calls": 0, "materialization_calls": 0, "sync_calls": 0}
        rec_fb = TestRecorder()
        with self.assertRaises(RuntimeError):
            execute_fallback(
                state,
                task,
                context,
                H=4,
                budget=budget,
                capabilities=FailingSampleCaps(),
                cfg=self.cfg,
                seed=42,
                clock_fn=lambda: 0.1,
                start_time=0.0,
                deadline=0.5,
                earliest_candidate=None,
                timing_counters=fb_counters,
                cache_counters={},
                audit_ids=(),
                recorder=rec_fb,
            )
        self.assertEqual(fb_counters["native_calls"], 1, "Fallback sample_prior attempt must be counted")
        self.assertEqual(fb_counters["sync_calls"], 1, "Fallback sync attempt must be counted")
        err_events = [e for e in rec_fb.events if e.get("name") == "operation.error"]
        self.assertTrue(len(err_events) > 0, "operation.error must be emitted on fallback failure")

        # 5. Op + sync double failure: op raises RuntimeError, sync raises ValueError
        # Must preserve original op exception (RuntimeError) and record sync_error in audit
        class DoubleFailCaps(DeterministicMockCapabilities):
            def synchronize(self):
                raise ValueError("sync blew up")

        rec_double = TestRecorder()
        double_counters = {"native_calls": 0, "sync_calls": 0}
        with self.assertRaises(RuntimeError) as cm:
            timed_operation(
                "sample_prior",
                lambda: (_ for _ in ()).throw(RuntimeError("primary op error")),
                clock_fn=lambda: 0.1,
                deadline=0.5,
                budget=budget,
                capabilities=DoubleFailCaps(),
                recorder=rec_double,
                timing_counters=double_counters,
                counter_key="native_calls",
            )
        self.assertEqual(str(cm.exception), "primary op error", "Must preserve primary op exception")
        self.assertEqual(double_counters["native_calls"], 1)
        self.assertEqual(double_counters["sync_calls"], 1)
        double_err_events = [e for e in rec_double.events if e.get("name") == "operation.error"]
        self.assertEqual(len(double_err_events), 1)
        self.assertIn("sync blew up", double_err_events[0]["fields"].get("sync_error", ""))

        # 6. Successful op then sync failure: emits operation.error and raises sync exception
        rec_sync_fail = TestRecorder()
        sync_fail_counters = {"native_calls": 0, "sync_calls": 0}
        with self.assertRaises(ValueError) as cm_sync:
            timed_operation(
                "sample_prior",
                lambda: "success_val",
                clock_fn=lambda: 0.1,
                deadline=0.5,
                budget=budget,
                capabilities=DoubleFailCaps(),
                recorder=rec_sync_fail,
                timing_counters=sync_fail_counters,
                counter_key="native_calls",
            )
        self.assertEqual(str(cm_sync.exception), "sync blew up")
        self.assertEqual(sync_fail_counters["native_calls"], 1)
        self.assertEqual(sync_fail_counters["sync_calls"], 1)
        sync_err_events = [e for e in rec_sync_fail.events if e.get("name") == "operation.error"]
        self.assertEqual(len(sync_err_events), 1)


if __name__ == "__main__":
    unittest.main()
