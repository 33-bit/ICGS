"""CPU checkpoint translation tests; no policy model or external artifact is loaded."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import warnings


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MISSING = [name for name in ("torch",) if importlib.util.find_spec(name) is None]
if not MISSING:
    import torch
    from torch import nn

    from ip.checkpoints import (
        CheckpointCompatibilityError,
        load_checkpoint_state,
        load_state_dict_compatible,
        normalize_state_dict,
    )
    from ip.utils.repairs import remove_prefix, repair_checkpoint


class TinyAGI(nn.Module if not MISSING else object):
    """Small modules with the same ownership names as AGI, but no runtime imports."""

    def __init__(self):
        super().__init__()
        self.graph = nn.Linear(2, 2, bias=False)
        self.scene_encoder = nn.Linear(2, 2, bias=False)
        self.local_encoder = nn.Linear(2, 2, bias=False)
        self.cond_encoder = nn.Linear(2, 2, bias=False)
        self.action_encoder = nn.Linear(2, 2, bias=False)
        self.prediction_head = nn.Linear(2, 2, bias=False)
        self.prediction_head_rot = nn.Linear(2, 2, bias=False)
        self.prediction_head_g = nn.Linear(2, 2, bias=False)


class TinyLegacyWrapper(nn.Module if not MISSING else object):
    def __init__(self):
        super().__init__()
        self.model = TinyAGI()
        self.graph_rep = self.model.graph
        self.scene_encoder = self.model.scene_encoder
        self.local_encoder = self.model.local_encoder
        self.cond_encoder = self.model.cond_encoder
        self.action_encoder = self.model.action_encoder
        self.action_head_trans = self.model.prediction_head
        self.action_head_rot = self.model.prediction_head_rot
        self.action_head_grip = self.model.prediction_head_g


@unittest.skipIf(bool(MISSING), "SKIPPED: missing CPU prerequisite: torch")
class CheckpointTests(unittest.TestCase):
    def test_raw_agi_state_synthesizes_expected_lightning_aliases(self):
        source = TinyAGI()
        target = TinyLegacyWrapper()
        for parameter in source.parameters():
            parameter.data.fill_(0.25)

        report = load_state_dict_compatible(target, source.state_dict())

        self.assertEqual(report.missing, ())
        self.assertEqual(report.unexpected, ())
        self.assertIn(("graph.weight", "model.graph.weight"), report.renamed)
        self.assertIn(("graph.weight", "graph_rep.weight"), report.renamed)
        torch.testing.assert_close(target.model.graph.weight, source.graph.weight)
        torch.testing.assert_close(target.graph_rep.weight, source.graph.weight)

    def test_legacy_lightning_aliases_collapse_into_raw_agi_state(self):
        source = TinyLegacyWrapper()
        target = TinyAGI()
        for parameter in source.parameters():
            parameter.data.fill_(0.75)

        report = load_state_dict_compatible(target, source.state_dict())

        self.assertEqual(report.conflicts, ())
        self.assertEqual(report.missing, ())
        self.assertEqual(report.unexpected, ())
        torch.testing.assert_close(target.graph.weight, source.model.graph.weight)
        torch.testing.assert_close(
            target.prediction_head_g.weight, source.model.prediction_head_g.weight
        )

    def test_compiled_markers_are_only_removed_as_literal_dotted_segments(self):
        expected = {
            "model.graph.weight": torch.zeros(2, 2),
            "not_orig_mod.weight": torch.zeros(1),
        }
        state = {
            "_orig_mod.model._orig_mod.graph.weight": torch.ones(2, 2),
            "not_orig_mod.weight": torch.ones(1),
        }

        translated, report = normalize_state_dict(state, expected)

        self.assertEqual(set(translated), set(expected))
        self.assertIn(
            ("_orig_mod.model._orig_mod.graph.weight", "model.graph.weight"),
            report.renamed,
        )
        self.assertNotIn(("not_orig_mod.weight", "weight"), report.renamed)

    def test_equal_duplicate_aliases_are_accepted_but_unequal_ones_conflict(self):
        expected = {"graph.weight": torch.zeros(2, 2)}
        equal = {
            "model.graph.weight": torch.ones(2, 2),
            "graph_rep.weight": torch.ones(2, 2),
        }
        translated, report = normalize_state_dict(equal, expected)
        torch.testing.assert_close(translated["graph.weight"], torch.ones(2, 2))
        self.assertEqual(report.conflicts, ())

        unequal = dict(equal)
        unequal["graph_rep.weight"] = torch.zeros(2, 2)
        with self.assertRaises(CheckpointCompatibilityError) as raised:
            normalize_state_dict(unequal, expected)
        self.assertTrue(raised.exception.report.conflicts)
        self.assertIn("model.graph.weight", str(raised.exception.report))
        self.assertIn("graph_rep.weight", str(raised.exception.report))

    def test_equal_collapsed_aliases_all_report_translation_when_exact_key_exists(self):
        expected = {"graph.weight": torch.zeros(2, 2)}
        state = {
            "graph.weight": torch.ones(2, 2),
            "graph_rep.weight": torch.ones(2, 2),
            "model._orig_mod.graph.weight": torch.ones(2, 2),
        }

        translated, report = normalize_state_dict(state, expected)

        torch.testing.assert_close(translated["graph.weight"], torch.ones(2, 2))
        self.assertEqual(
            report.renamed,
            (
                ("graph_rep.weight", "graph.weight"),
                ("model._orig_mod.graph.weight", "graph.weight"),
            ),
        )

    def test_strict_missing_and_unexpected_fail_before_mutating_module(self):
        module = nn.Linear(2, 2)
        before = module.weight.detach().clone()
        state = {
            "weight": torch.full_like(module.weight, 3),
            "extra": torch.ones(1),
        }

        with self.assertRaises(CheckpointCompatibilityError) as raised:
            load_state_dict_compatible(module, state, strict=True)

        self.assertEqual(raised.exception.report.missing, ("bias",))
        self.assertEqual(raised.exception.report.unexpected, ("extra",))
        torch.testing.assert_close(module.weight, before)

    def test_non_strict_load_warns_with_full_report_and_loads_compatible_values(self):
        module = nn.Linear(2, 2)
        state = {
            "weight": torch.full_like(module.weight, 4),
            "extra": torch.ones(1),
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = load_state_dict_compatible(module, state, strict=False)

        self.assertEqual(len(caught), 1)
        self.assertIn(repr(report), str(caught[0].message))
        self.assertEqual(report.missing, ("bias",))
        self.assertEqual(report.unexpected, ("extra",))
        torch.testing.assert_close(module.weight, torch.full_like(module.weight, 4))

    def test_shape_mismatch_always_fails_even_in_non_strict_mode(self):
        module = nn.Linear(2, 2, bias=False)
        with self.assertRaises(CheckpointCompatibilityError) as raised:
            load_state_dict_compatible(
                module, {"weight": torch.ones(3, 2)}, strict=False
            )
        self.assertTrue(raised.exception.report.shape_mismatches)
        self.assertIn("(3, 2)", str(raised.exception.report))
        self.assertIn("(2, 2)", str(raised.exception.report))

    def test_trusted_lightning_artifact_load_does_not_rewrite_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lightning.ckpt"
            module = nn.Linear(2, 2)
            checkpoint = {
                "state_dict": {
                    "_orig_mod.weight": torch.full_like(module.weight, 5),
                    "_orig_mod.bias": torch.full_like(module.bias, 6),
                },
                "epoch": 17,
                "callbacks": {"example": {"best": 0.1}},
            }
            torch.save(checkpoint, path)
            before = path.read_bytes()

            report = load_checkpoint_state(path, module)

            self.assertEqual(report.missing, ())
            self.assertEqual(path.read_bytes(), before)
            torch.testing.assert_close(module.weight, torch.full_like(module.weight, 5))
            torch.testing.assert_close(module.bias, torch.full_like(module.bias, 6))

    def test_trusted_raw_artifact_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.pt"
            module = nn.Linear(2, 2, bias=False)
            torch.save({"weight": torch.full_like(module.weight, 8)}, path)

            report = load_checkpoint_state(path, module)

            self.assertEqual(report.missing, ())
            torch.testing.assert_close(module.weight, torch.full_like(module.weight, 8))

    def test_repair_requires_output_preserves_metadata_and_leaves_source_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.ckpt"
            output = Path(directory) / "repaired.ckpt"
            checkpoint = {
                "state_dict": {"model._orig_mod.layer.weight": torch.ones(2)},
                "epoch": 9,
                "optimizer_states": [{"step": 3}],
            }
            torch.save(checkpoint, source)
            before = source.read_bytes()

            with self.assertRaisesRegex(ValueError, "save_path"):
                repair_checkpoint(source)
            repair_checkpoint(source, output)

            self.assertEqual(source.read_bytes(), before)
            repaired = torch.load(output, weights_only=False)
            self.assertEqual(repaired["epoch"], 9)
            self.assertEqual(repaired["optimizer_states"], [{"step": 3}])
            self.assertEqual(set(repaired["state_dict"]), {"model.layer.weight"})

    def test_repair_skips_write_when_unneeded_and_rejects_unequal_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            clean = Path(directory) / "clean.ckpt"
            unused_output = Path(directory) / "unused.ckpt"
            torch.save({"state_dict": {"layer.weight": torch.ones(2)}}, clean)
            self.assertIsNone(repair_checkpoint(clean, unused_output))
            self.assertFalse(unused_output.exists())

            collision = Path(directory) / "collision.ckpt"
            torch.save(
                {
                    "state_dict": {
                        "layer._orig_mod.weight": torch.ones(2),
                        "layer.weight": torch.zeros(2),
                    }
                },
                collision,
            )
            with self.assertRaisesRegex(ValueError, "collision"):
                repair_checkpoint(collision, Path(directory) / "out.ckpt")

    def test_remove_prefix_is_leading_and_literal(self):
        self.assertEqual(remove_prefix("model.weight", "model."), "weight")
        self.assertEqual(remove_prefix("other.model.weight", "model."), "other.model.weight")
        self.assertEqual(remove_prefix("modelXweight", "model."), "modelXweight")


if __name__ == "__main__":
    unittest.main()
