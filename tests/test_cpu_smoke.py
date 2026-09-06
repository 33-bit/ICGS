"""L1 characterization only: no model, simulator, dataset, GPU, or downloads."""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
import unittest


# Run from any working directory without requiring package installation.
MISSING = [name for name in ("torch", "numpy") if importlib.util.find_spec(name) is None]
if not MISSING:
    # Do not swallow errors from packages that are installed but broken.
    import torch
    from icgs.configuration.defaults import instant_policy_original, to_legacy
    config = to_legacy(instant_policy_original())
    from icgs.algorithms.diffusion.codec import Normalizer


@unittest.skipIf(bool(MISSING), "SKIPPED: missing CPU prerequisites: " + ", ".join(MISSING))
class CpuSmokeTests(unittest.TestCase):
    def make_normalizer(self):
        return Normalizer(
            pred_horizon=2,
            min_action=torch.tensor([-0.01] * 3 + [-math.pi / 60] * 3),
            max_action=torch.tensor([0.01] * 3 + [math.pi / 60] * 3),
            device="cpu",
        )

    def test_derived_config_does_not_mutate_baseline_tensors(self):
        before = config["min_actions"].clone()
        original_demos = config["num_demos"]
        derived = deepcopy(config)
        derived["min_actions"].zero_()
        derived["num_demos"] = original_demos + 1
        torch.testing.assert_close(config["min_actions"], before)
        self.assertEqual(config["num_demos"], original_demos)
        self.assertEqual(config["min_actions"].shape, (6,))
        self.assertTrue(torch.isfinite(config["min_actions"]).all())

    def test_horizon_bounds_have_independent_numeric_expectations(self):
        normalizer = self.make_normalizer()
        expected = torch.tensor([[[0.01] * 3 + [math.pi / 60] * 3,
                                  [0.02] * 3 + [math.pi / 30] * 3]])
        torch.testing.assert_close(normalizer.max_action, expected)
        torch.testing.assert_close(normalizer.min_action, -expected)
        torch.testing.assert_close(normalizer.normalize_actions(expected), torch.ones(1, 2, 6))
        torch.testing.assert_close(normalizer.normalize_actions(-expected), -torch.ones(1, 2, 6))
        # Second-horizon displacement bound from geometry, independently calculated.
        displacement = 2 * math.sqrt(2 * 0.06**2 * (1 - math.cos(math.pi / 30)))
        self.assertAlmostEqual(normalizer.max_labels[0, 1, 0, 0].item(), 0.04, places=6)
        self.assertAlmostEqual(normalizer.max_labels[0, 1, 0, 3].item(), displacement, places=6)

    def test_action_round_trip_with_nonzero_translation_and_rotation(self):
        normalizer = self.make_normalizer()
        values = torch.tensor([[[0.003, -0.004, 0.005, 0.01, -0.02, 0.03],
                                [-0.011, 0.012, 0.001, -0.03, 0.02, 0.04]]])
        torch.testing.assert_close(normalizer.denormalize_actions(normalizer.normalize_actions(values)), values)

    def test_label_round_trip_and_zero_center(self):
        normalizer = self.make_normalizer()
        labels = torch.tensor([[[[0.003, -0.004, 0.005, 0.001, -0.002, 0.003]],
                                [[0.006, -0.007, 0.008, 0.003, -0.004, 0.005]]]])
        torch.testing.assert_close(normalizer.denormalize_labels(normalizer.normalize_labels(labels)), labels)
        torch.testing.assert_close(normalizer.normalize_labels(torch.zeros_like(labels)), torch.zeros_like(labels))


if __name__ == "__main__":
    unittest.main()
