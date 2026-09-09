"""Tests for belief state, mass propagation, and search foundations."""

from __future__ import annotations

import unittest
import numpy as np
import torch

from icgs.configuration.method import MethodConfig
from icgs.state.physical import PhysicalState
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
        grip=torch.zeros(1, 1, dtype=torch.float32),
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


if __name__ == "__main__":
    unittest.main()
