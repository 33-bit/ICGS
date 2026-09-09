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

        # Root choice broke to Edge 0 (most visits: 2 > 1)
        self.assertEqual(res.selected_prefix.raw_candidate_id, "cand_0")

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

    def test_audit_issue_2_bounded_budget_and_operation_level_cap_interrupt(self) -> None:
        """PlanningBudget requires at least one bound; cap reached during edge stepping interrupts without evaluating incomplete edge."""
        from icgs.algorithms.planning.budget import PlanningBudget
        from icgs.algorithms.planning.mcts import plan

        # Budget requires at least one bound
        with self.assertRaises(ValueError):
            PlanningBudget()

        caps = DeterministicMockCapabilities()
        state = _make_physical_state(translation=(0.0, 0.0, 0.0), boundary=0)
        task = {"history_id": "root_task", "step": 0}
        context = {"context_id": "ctx_0"}

        # Edge needs 2 steps * 3 heads = 6 model intervals.
        # Cap at 1 model interval: first predict_step happens, then cap reached.
        # Incomplete edge must NOT be evaluated, NOT added to tree, NOT backed up.
        budget = PlanningBudget(model_interval_cap=1)
        result = plan(state, task, context, H=4, budget=budget, capabilities=caps, cfg=self.cfg)

        self.assertFalse(result.completed)
        self.assertIsNone(result.selected_prefix)
        self.assertEqual(result.fallback_reason, "budget_exhausted")
        self.assertEqual(result.timing_counters["model_intervals"], 1)
        self.assertEqual(len(caps.predict_step_calls), 1)

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
        self.assertEqual(obs.points.shape[-1], 3)
        self.assertEqual(obs.T_w_e.shape, (4, 4))
        self.assertIn(obs.grip, (0.0, 1.0))

    def test_audit_issue_4_config_discipline_and_no_duck_typing(self) -> None:
        """MethodConfig/PlanningConfig is strictly required; duck typing and legacy tunables are rejected."""
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


if __name__ == "__main__":
    unittest.main()
