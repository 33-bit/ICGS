import ast
import inspect
import random
import unittest
from dataclasses import replace

import numpy as np
import torch

from icgs.configuration.method import MethodConfig


class TaskTrackerTensorTests(unittest.TestCase):
    @staticmethod
    def _inputs(*, batch=1):
        width = MethodConfig().event.width
        physical_tokens = torch.linspace(
            -0.5, 0.5, batch * 5 * width, dtype=torch.float32
        ).reshape(batch, 5, width)
        event_tokens = torch.linspace(
            0.25, 0.75, batch * 5 * width, dtype=torch.float32
        ).reshape(batch, 5, width)
        physical_valid = torch.tensor(
            [[True, True, True, True, False]] * batch,
            dtype=torch.bool,
        )
        event_valid = torch.tensor(
            [[True, True, True, True, False]] * batch,
            dtype=torch.bool,
        )
        previous_r = torch.linspace(-0.1, 0.1, batch * width).reshape(batch, width)
        return {
            "physical_tokens": physical_tokens,
            "physical_valid": physical_valid,
            "event_tokens": event_tokens,
            "event_valid": event_valid,
            "previous_r": previous_r,
        }

    @staticmethod
    def _tampered_config(section, **changes):
        config = MethodConfig()
        object.__setattr__(config, section, replace(getattr(config, section), **changes))
        return config

    def test_task_event_features_are_exact_and_strictly_finite(self):
        from icgs.models.memories.task import task_event_features

        events = torch.ones(1, 4, 256)
        previous = torch.arange(256, dtype=torch.float32)[None]
        features = task_event_features(events, previous)
        self.assertEqual(tuple(features.shape), (1, 4, 768))
        torch.testing.assert_close(features[..., :256], events)
        torch.testing.assert_close(
            features[..., 256:512], previous[:, None].expand_as(events)
        )
        torch.testing.assert_close(features[..., 512:], events * previous[:, None])

        invalid_events = events.clone()
        invalid_events[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            task_event_features(invalid_events, previous)
        invalid_previous = previous.clone()
        invalid_previous[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            task_event_features(events, invalid_previous)

    def test_generic_cross_attention_sanitizes_both_masks(self):
        from icgs.models.layers.method import MaskedCrossAttentionBlock

        config = MethodConfig()
        width = config.event.width
        block = MaskedCrossAttentionBlock(
            width=width,
            neural_config=config.neural,
        ).eval()
        self.assertIsNot(block.norm_query, block.norm_keys)
        self.assertFalse(hasattr(block, "geometry_bias"))

        query_valid = torch.tensor([[True, False]], dtype=torch.bool)
        key_valid = torch.tensor([[True, False, True]], dtype=torch.bool)
        query = torch.randn(1, 2, width)
        keys = torch.randn(1, 3, width)
        clean_query = query.masked_fill(~query_valid[..., None], 0.0)
        clean_keys = keys.masked_fill(~key_valid[..., None], 0.0)
        poisoned_query = clean_query.clone()
        poisoned_keys = clean_keys.clone()
        poisoned_query[0, 1] = float("nan")
        poisoned_keys[0, 1] = float("inf")
        with torch.no_grad():
            clean = block(clean_query, query_valid, clean_keys, key_valid)
            poisoned = block(poisoned_query, query_valid, poisoned_keys, key_valid)
        torch.testing.assert_close(poisoned, clean)
        torch.testing.assert_close(poisoned[:, 1], torch.zeros_like(poisoned[:, 1]))

        bad_query = clean_query.clone()
        bad_query[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            block(bad_query, query_valid, clean_keys, key_valid)
        bad_keys = clean_keys.clone()
        bad_keys[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            block(clean_query, query_valid, bad_keys, key_valid)
        with self.assertRaisesRegex(ValueError, "valid key"):
            block(clean_query, query_valid, clean_keys, torch.zeros_like(key_valid))

    def test_tracker_requires_explicit_compatible_configuration(self):
        from icgs.models.memories.task import TaskTracker

        with self.assertRaises(TypeError):
            TaskTracker()
        with self.assertRaisesRegex(TypeError, "MethodConfig"):
            TaskTracker(object())
        tracker = TaskTracker(MethodConfig())
        self.assertEqual(len(tracker.cross_attention_blocks), 2)
        self.assertEqual(
            (
                tracker.scene_projection.in_features,
                tracker.scene_projection.out_features,
            ),
            (256, 256),
        )
        self.assertEqual(
            (tracker.event_head[0].in_features, tracker.event_head[0].out_features),
            (768, 256),
        )
        self.assertFalse(
            any(
                isinstance(module, (torch.nn.Sigmoid, torch.nn.Softmax))
                for module in tracker.modules()
            )
        )

        invalid_configs = (
            (self._tampered_config("event", width=128), "width"),
            (self._tampered_config("memory", width=128), "width"),
            (self._tampered_config("tracker", memory_slots=2), "memory_slots"),
            (self._tampered_config("tracker", attention_layers=0), "attention_layers"),
            (self._tampered_config("neural", attention_heads=7), "divide|heads"),
        )
        for config, message in invalid_configs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    TaskTracker(config)

    def test_tracker_rejects_invalid_shapes_dtypes_and_empty_scene(self):
        from icgs.models.memories.task import TaskTracker

        tracker = TaskTracker(MethodConfig())
        inputs = self._inputs()
        invalid_cases = []
        wrong_physical = {name: value.clone() for name, value in inputs.items()}
        wrong_physical["physical_tokens"] = wrong_physical["physical_tokens"][:, :, :-1]
        invalid_cases.append((wrong_physical, "physical_tokens"))
        wrong_events = {name: value.clone() for name, value in inputs.items()}
        wrong_events["event_tokens"] = wrong_events["event_tokens"][:, :, :-1]
        invalid_cases.append((wrong_events, "event_tokens"))
        wrong_previous = {name: value.clone() for name, value in inputs.items()}
        wrong_previous["previous_r"] = wrong_previous["previous_r"][:, :-1]
        invalid_cases.append((wrong_previous, "previous_r"))
        wrong_mask = {name: value.clone() for name, value in inputs.items()}
        wrong_mask["event_valid"] = wrong_mask["event_valid"].to(torch.int64)
        invalid_cases.append((wrong_mask, "bool"))
        mixed_dtype = {name: value.clone() for name, value in inputs.items()}
        mixed_dtype["event_tokens"] = mixed_dtype["event_tokens"].to(torch.float64)
        invalid_cases.append((mixed_dtype, "dtype"))
        empty_scene = {name: value.clone() for name, value in inputs.items()}
        empty_scene["physical_valid"].zero_()
        invalid_cases.append((empty_scene, "valid physical"))

        for invalid, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    tracker(**invalid)

    def test_tracker_tensor_surface_and_masked_output_contract(self):
        import icgs.models.memories.task as task_module
        from icgs.models.memories.task import TaskEncoding, TaskTracker

        parameters = tuple(inspect.signature(TaskTracker.forward).parameters)[1:]
        self.assertEqual(
            parameters,
            (
                "physical_tokens",
                "physical_valid",
                "event_tokens",
                "event_valid",
                "previous_r",
            ),
        )
        self.assertEqual(
            tuple(TaskEncoding.__dataclass_fields__),
            ("r", "alignment_logits", "event_logits", "event_valid"),
        )
        source = inspect.getsource(task_module)
        for forbidden in (
            "TaskState",
            "MethodContext",
            "SegmentRef",
            "InstantPolicy",
            "router",
            "program_id",
            "algorithms.objectives",
        ):
            self.assertNotIn(forbidden, source)

        config = MethodConfig()
        tracker = TaskTracker(config).eval()
        inputs = self._inputs()
        with torch.no_grad():
            encoded = tracker(**inputs)
        self.assertIsInstance(encoded, TaskEncoding)
        self.assertEqual(tuple(encoded.r.shape), (1, config.event.width))
        self.assertEqual(tuple(encoded.alignment_logits.shape), (1, 6))
        self.assertEqual(tuple(encoded.event_logits.shape), (1, 5, 3))
        self.assertIs(encoded.event_valid, inputs["event_valid"])
        self.assertTrue(torch.isfinite(encoded.r).all().item())
        self.assertTrue(torch.isfinite(encoded.alignment_logits).all().item())
        self.assertTrue(torch.isfinite(encoded.event_logits).all().item())
        self.assertEqual(
            encoded.alignment_logits[0, 4].item(),
            torch.finfo(encoded.alignment_logits.dtype).min,
        )
        self.assertTrue(torch.isfinite(encoded.alignment_logits[:, -1]).all().item())
        torch.testing.assert_close(
            encoded.event_logits[:, 4],
            torch.zeros_like(encoded.event_logits[:, 4]),
        )

    def test_tracker_rejects_valid_nonfinite_and_sanitizes_padding(self):
        from icgs.models.memories.task import TaskTracker

        tracker = TaskTracker(MethodConfig()).eval()
        clean = self._inputs()
        clean["physical_tokens"] = clean["physical_tokens"].masked_fill(
            ~clean["physical_valid"][..., None], 0.0
        )
        clean["event_tokens"] = clean["event_tokens"].masked_fill(
            ~clean["event_valid"][..., None], 0.0
        )
        poisoned = {name: value.clone() for name, value in clean.items()}
        poisoned["physical_tokens"][0, 4] = float("nan")
        poisoned["event_tokens"][0, 4] = float("inf")
        with torch.no_grad():
            clean_output = tracker(**clean)
            poisoned_output = tracker(**poisoned)
        torch.testing.assert_close(poisoned_output.r, clean_output.r)
        torch.testing.assert_close(
            poisoned_output.alignment_logits, clean_output.alignment_logits
        )
        torch.testing.assert_close(poisoned_output.event_logits, clean_output.event_logits)

        for name, index in (
            ("physical_tokens", (0, 0, 0)),
            ("event_tokens", (0, 0, 0)),
            ("previous_r", (0, 0)),
        ):
            with self.subTest(name=name):
                invalid = {key: value.clone() for key, value in clean.items()}
                invalid[name][index] = float("nan")
                with self.assertRaisesRegex(ValueError, "finite"):
                    tracker(**invalid)

    def test_scene_mean_includes_all_valid_physical_rows(self):
        from icgs.models.memories.task import TaskTracker
        from icgs.models.layers.method import masked_mean

        tracker = TaskTracker(MethodConfig()).eval()
        inputs = self._inputs()
        captured = []

        def capture_scene(_module, arguments):
            captured.append(arguments[0].detach().clone())

        handle = tracker.scene_projection.register_forward_pre_hook(capture_scene)
        try:
            with torch.no_grad():
                tracker(**inputs)
        finally:
            handle.remove()
        self.assertEqual(len(captured), 1)
        torch.testing.assert_close(
            captured[0],
            masked_mean(inputs["physical_tokens"], inputs["physical_valid"], dim=1),
        )

    def test_demo_block_permutation_is_equivariant(self):
        from icgs.models.memories.task import TaskTracker

        torch.manual_seed(23)
        tracker = TaskTracker(MethodConfig()).eval()
        inputs = self._inputs()
        inputs["event_valid"] = torch.ones_like(inputs["event_valid"])
        permutation = torch.tensor([2, 3, 0, 1, 4], dtype=torch.long)
        permuted = {name: value.clone() for name, value in inputs.items()}
        permuted["event_tokens"] = inputs["event_tokens"][:, permutation]
        permuted["event_valid"] = inputs["event_valid"][:, permutation]
        with torch.no_grad():
            original = tracker(**inputs)
            swapped = tracker(**permuted)
        torch.testing.assert_close(swapped.r, original.r, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            swapped.alignment_logits[:, :-1],
            original.alignment_logits[:, permutation],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            swapped.alignment_logits[:, -1],
            original.alignment_logits[:, -1],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            swapped.event_logits,
            original.event_logits[:, permutation],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_all_invalid_events_leave_only_finite_null_alignment(self):
        from icgs.models.memories.task import TaskTracker

        tracker = TaskTracker(MethodConfig()).eval()
        inputs = self._inputs()
        inputs["event_valid"].zero_()
        inputs["event_tokens"].fill_(float("nan"))
        with torch.no_grad():
            encoded = tracker(**inputs)
        sentinel = torch.finfo(encoded.alignment_logits.dtype).min
        torch.testing.assert_close(
            encoded.alignment_logits[:, :-1],
            torch.full_like(encoded.alignment_logits[:, :-1], sentinel),
        )
        self.assertTrue(torch.isfinite(encoded.alignment_logits[:, -1]).all().item())
        torch.testing.assert_close(
            encoded.event_logits,
            torch.zeros_like(encoded.event_logits),
        )
        torch.testing.assert_close(
            torch.softmax(encoded.alignment_logits, dim=-1)[:, -1],
            torch.ones(encoded.alignment_logits.shape[0]),
        )

    def test_tracker_keeps_gradients_to_inputs_and_parameters(self):
        from icgs.models.memories.task import TaskTracker

        tracker = TaskTracker(MethodConfig()).eval()
        inputs = self._inputs()
        for name in ("physical_tokens", "event_tokens", "previous_r"):
            inputs[name].requires_grad_(True)
        encoded = tracker(**inputs)
        loss = (
            encoded.r.square().mean()
            + encoded.alignment_logits[:, :-1][encoded.event_valid].square().mean()
            + encoded.alignment_logits[:, -1].square().mean()
            + encoded.event_logits[encoded.event_valid].square().mean()
        )
        loss.backward()
        for name in ("physical_tokens", "event_tokens", "previous_r"):
            gradient = inputs[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all().item(), name)
        torch.testing.assert_close(
            inputs["physical_tokens"].grad[~inputs["physical_valid"]],
            torch.zeros_like(
                inputs["physical_tokens"].grad[~inputs["physical_valid"]]
            ),
        )
        torch.testing.assert_close(
            inputs["event_tokens"].grad[~inputs["event_valid"]],
            torch.zeros_like(inputs["event_tokens"].grad[~inputs["event_valid"]]),
        )
        for name, parameter in tracker.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)


class TaskStateOwnershipTests(unittest.TestCase):
    @staticmethod
    def _events(*, batch=1):
        from icgs.contracts.method import SegmentRef
        from icgs.models.encoders.event import EventEncoding
        from icgs.state.method_context import build_event_memory

        width = MethodConfig().event.width
        tokens = torch.linspace(
            -0.25,
            0.75,
            batch * 3 * width,
            dtype=torch.float32,
        ).reshape(batch, 3, width)
        valid = torch.tensor([[True, True, False]] * batch, dtype=torch.bool)
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        hashes = tuple((f"demo-{row}",) for row in range(batch))
        refs = tuple(
            (
                SegmentRef(hashes[row][0], 0, 0, "start", False),
                SegmentRef(hashes[row][0], 0, 1, "interaction", True),
                None,
            )
            for row in range(batch)
        )
        return build_event_memory(
            EventEncoding(tokens, valid),
            refs,
            hashes,
            "event-encoder-v1",
            "segmentation-v1",
        )

    @staticmethod
    def _encoding(events, *, requires_grad=False):
        from icgs.models.memories.task import TaskEncoding

        batch, event_count, width = events.tokens.shape
        r = torch.linspace(-0.2, 0.2, batch * width).reshape(batch, width)
        alignment_logits = torch.tensor(
            [[0.0, 1.0, torch.finfo(torch.float32).min, -0.5]] * batch,
            dtype=torch.float32,
        )
        event_logits = torch.tensor(
            [[[8.0, -8.0, 0.5], [-8.0, 8.0, -0.5], [0.0, 0.0, 0.0]]] * batch,
            dtype=torch.float32,
        )
        if event_count != 3:
            raise AssertionError("fixture expects exactly three event rows")
        if requires_grad:
            r.requires_grad_(True)
            alignment_logits.requires_grad_(True)
            event_logits.requires_grad_(True)
        return TaskEncoding(r, alignment_logits, event_logits, events.valid.clone())

    @staticmethod
    def _direct_state(*, boundary=3, tracker_id="tracker-v1"):
        from icgs.state.task import TaskState

        return TaskState(
            r=torch.zeros(1, 256),
            alpha=torch.tensor([[0.2, 0.3, 0.0, 0.5]]),
            rho=torch.tensor([[1.0, 0.1, 0.0]]),
            nu=torch.tensor([[0.0, 0.9, 0.0]]),
            eligible=torch.tensor([[0.7, 0.2, 0.0]]),
            boundary=boundary,
            context_fingerprints=("context-a",),
            tracker_id=tracker_id,
        )

    def test_initial_task_memory_matches_event_tensor_owner(self):
        from icgs.state.task import initial_task_memory

        events = self._events(batch=2)
        initial = initial_task_memory(events)
        self.assertEqual(tuple(initial.shape), (2, 256))
        self.assertEqual(initial.dtype, events.tokens.dtype)
        self.assertEqual(initial.device, events.tokens.device)
        torch.testing.assert_close(initial, torch.zeros_like(initial))
        with self.assertRaisesRegex(TypeError, "EventMemory"):
            initial_task_memory(object())

    def test_builder_materializes_probabilities_and_exact_padding_zeros(self):
        from icgs.state.task import build_task_state

        events = self._events()
        encoding = self._encoding(events)
        state = build_task_state(
            encoding,
            events,
            boundary=4,
            tracker_id="tracker-v1",
        )
        expected_alpha = torch.softmax(encoding.alignment_logits, dim=-1)
        torch.testing.assert_close(state.alpha, expected_alpha)
        torch.testing.assert_close(
            state.rho[:, :2], torch.sigmoid(encoding.event_logits[:, :2, 0])
        )
        torch.testing.assert_close(
            state.nu[:, :2], torch.sigmoid(encoding.event_logits[:, :2, 1])
        )
        torch.testing.assert_close(
            state.eligible[:, :2], torch.sigmoid(encoding.event_logits[:, :2, 2])
        )
        torch.testing.assert_close(state.alpha[:, 2], torch.zeros_like(state.alpha[:, 2]))
        for value in (state.rho, state.nu, state.eligible):
            torch.testing.assert_close(value[:, 2], torch.zeros_like(value[:, 2]))
        self.assertEqual(state.boundary, 4)

    def test_masked_softmax_all_invalid_events_leaves_only_null(self):
        from icgs.state.task import _masked_softmax_with_null

        logits = torch.tensor([[2.0, -3.0, 0.75]], dtype=torch.float32)
        valid = torch.zeros(1, 2, dtype=torch.bool)
        alpha = _masked_softmax_with_null(logits, valid)
        torch.testing.assert_close(alpha, torch.tensor([[0.0, 0.0, 1.0]]))

    def test_rho_nu_and_eligible_are_independent(self):
        from icgs.state.task import build_task_state

        events = self._events()
        state = build_task_state(
            self._encoding(events), events, boundary=0, tracker_id="tracker-v1"
        )
        self.assertGreater(state.rho[0, 0].item(), 0.99)
        self.assertLess(state.nu[0, 0].item(), 0.01)
        self.assertNotEqual(state.eligible[0, 0].item(), state.nu[0, 0].item())

    def test_task_state_rejects_incompatible_inner_shapes(self):
        from icgs.state.task import TaskState

        with self.assertRaisesRegex(ValueError, "rho"):
            TaskState(
                r=torch.zeros(1, 256),
                alpha=torch.tensor([[0.2, 0.3, 0.0, 0.5]]),
                rho=torch.zeros(1, 2),
                nu=torch.zeros(1, 3),
                eligible=torch.zeros(1, 3),
                boundary=0,
                context_fingerprints=("context-a",),
                tracker_id="tracker-v1",
            )

    def test_task_state_rejects_nonfinite_out_of_range_and_bad_alpha_sum(self):
        from icgs.state.task import TaskState

        base = self._direct_state()
        invalid = (
            ("rho", torch.tensor([[float("nan"), 0.0, 0.0]]), "finite"),
            ("nu", torch.tensor([[1.1, 0.0, 0.0]]), r"\[0,1\]"),
            ("eligible", torch.tensor([[-0.1, 0.0, 0.0]]), r"\[0,1\]"),
            ("alpha", torch.tensor([[0.2, 0.3, 0.0, 0.4]]), "sum"),
        )
        for field, value, message in invalid:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    TaskState(
                        r=base.r,
                        alpha=value if field == "alpha" else base.alpha,
                        rho=value if field == "rho" else base.rho,
                        nu=value if field == "nu" else base.nu,
                        eligible=value if field == "eligible" else base.eligible,
                        boundary=base.boundary,
                        context_fingerprints=base.context_fingerprints,
                        tracker_id=base.tracker_id,
                    )
        near_normalized = base.alpha.clone()
        near_normalized[0, -1] += torch.finfo(near_normalized.dtype).eps
        TaskState(
            base.r,
            near_normalized,
            base.rho,
            base.nu,
            base.eligible,
            base.boundary,
            base.context_fingerprints,
            base.tracker_id,
        )

    def test_task_state_validates_boundary_and_lineage_identifiers(self):
        from icgs.state.task import TaskState

        base = self._direct_state()
        cases = (
            ({"boundary": -1}, "boundary"),
            ({"boundary": True}, "boundary"),
            ({"context_fingerprints": ["context-a"]}, "tuple"),
            ({"context_fingerprints": ("",)}, "nonempty"),
            ({"context_fingerprints": ()}, "batch"),
            ({"tracker_id": "  "}, "nonempty"),
        )
        for changes, message in cases:
            values = {
                "r": base.r,
                "alpha": base.alpha,
                "rho": base.rho,
                "nu": base.nu,
                "eligible": base.eligible,
                "boundary": base.boundary,
                "context_fingerprints": base.context_fingerprints,
                "tracker_id": base.tracker_id,
            }
            values.update(changes)
            with self.subTest(changes=changes):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    TaskState(**values)

    def test_builder_rejects_mask_or_shape_mismatch(self):
        from icgs.models.memories.task import TaskEncoding
        from icgs.state.task import build_task_state

        events = self._events()
        encoding = self._encoding(events)
        mismatched_mask = encoding.event_valid.clone()
        mismatched_mask[0, 1] = False
        cases = (
            (replace(encoding, event_valid=mismatched_mask), "event_valid"),
            (replace(encoding, r=encoding.r[:, :-1]), "r"),
            (
                replace(
                    encoding,
                    alignment_logits=encoding.alignment_logits[:, :-1],
                ),
                "alignment_logits",
            ),
            (replace(encoding, event_logits=encoding.event_logits[:, :-1]), "event_logits"),
        )
        for invalid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_task_state(invalid, events, boundary=0, tracker_id="tracker-v1")

    def test_builder_rejects_invalid_raw_padding_contract(self):
        from icgs.state.task import build_task_state

        events = self._events()
        encoding = self._encoding(events)
        bad_alignment = encoding.alignment_logits.clone()
        bad_alignment[0, 2] = 0.0
        bad_events = encoding.event_logits.clone()
        bad_events[0, 2, 0] = 1.0
        for invalid, message in (
            (replace(encoding, alignment_logits=bad_alignment), "alignment"),
            (replace(encoding, event_logits=bad_events), "event logits"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_task_state(invalid, events, boundary=0, tracker_id="tracker-v1")

    def test_builder_derives_ordered_context_lineage_and_has_narrow_surface(self):
        import icgs.state.task as task_module
        from icgs.state.task import TaskState, build_task_state

        events = self._events(batch=2)
        state = build_task_state(
            self._encoding(events), events, boundary=1, tracker_id="tracker-v1"
        )
        self.assertEqual(state.context_fingerprints, events.fingerprints)
        self.assertNotIn("event_valid", TaskState.__dataclass_fields__)
        self.assertNotIn(
            "context_fingerprints", inspect.signature(build_task_state).parameters
        )
        source = inspect.getsource(task_module)
        for forbidden in (
            "track_task",
            "InstantPolicy",
            "PhysicalState",
            "router",
            "algorithms.objectives",
            "program_id",
        ):
            self.assertNotIn(forbidden, source)

    def test_builder_owns_tensors_without_detaching_autograd(self):
        from icgs.state.task import build_task_state

        events = self._events()
        encoding = self._encoding(events, requires_grad=True)
        state = build_task_state(
            encoding, events, boundary=2, tracker_id="tracker-v1"
        )
        self.assertNotEqual(
            state.r.untyped_storage().data_ptr(),
            encoding.r.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            state.rho.untyped_storage().data_ptr(),
            encoding.event_logits.untyped_storage().data_ptr(),
        )
        before = state.r.clone()
        with torch.no_grad():
            encoding.r.add_(10.0)
        torch.testing.assert_close(state.r, before)
        alpha_weight = torch.arange(
            state.alpha.shape[-1],
            dtype=state.alpha.dtype,
            device=state.alpha.device,
        )
        loss = (
            state.r.sum()
            + (state.alpha * alpha_weight).sum()
            + state.rho.sum()
            + 2.0 * state.nu.sum()
            + 3.0 * state.eligible.sum()
        )
        loss.backward()
        self.assertIsNotNone(encoding.r.grad)
        self.assertIsNotNone(encoding.alignment_logits.grad)
        self.assertIsNotNone(encoding.event_logits.grad)
        self.assertTrue(torch.isfinite(encoding.r.grad).all().item())
        self.assertTrue(torch.isfinite(encoding.alignment_logits.grad).all().item())
        self.assertTrue(torch.isfinite(encoding.event_logits.grad).all().item())
        self.assertTrue(
            (encoding.alignment_logits.grad[:, :-1][events.valid] != 0).any().item()
        )
        self.assertTrue((encoding.event_logits.grad[events.valid] != 0).all().item())
        torch.testing.assert_close(
            encoding.alignment_logits.grad[:, :-1][~events.valid],
            torch.zeros_like(
                encoding.alignment_logits.grad[:, :-1][~events.valid]
            ),
        )
        torch.testing.assert_close(
            encoding.event_logits.grad[~events.valid],
            torch.zeros_like(encoding.event_logits.grad[~events.valid]),
        )

    def test_branch_copy_is_storage_independent(self):
        state = self._direct_state()
        branch = state.branch_copy()
        self.assertEqual(branch.boundary, state.boundary)
        self.assertEqual(branch.context_fingerprints, state.context_fingerprints)
        for name in ("r", "alpha", "rho", "nu", "eligible"):
            left = getattr(state, name)
            right = getattr(branch, name)
            torch.testing.assert_close(left, right)
            self.assertNotEqual(
                left.untyped_storage().data_ptr(), right.untyped_storage().data_ptr()
            )

    def test_task_update_requires_shape_lineage_and_exact_successor(self):
        from icgs.state.task import TaskState, validate_task_update

        previous = self._direct_state(boundary=7)
        current = self._direct_state(boundary=8)
        self.assertIsNone(validate_task_update(previous, current))
        cases = (
            (self._direct_state(boundary=7), "successor"),
            (self._direct_state(boundary=9), "successor"),
            (self._direct_state(boundary=8, tracker_id="tracker-v2"), "tracker"),
            (
                TaskState(
                    current.r,
                    current.alpha,
                    current.rho,
                    current.nu,
                    current.eligible,
                    current.boundary,
                    ("context-b",),
                    current.tracker_id,
                ),
                "context",
            ),
            (
                TaskState(
                    torch.zeros(1, 128),
                    current.alpha,
                    current.rho,
                    current.nu,
                    current.eligible,
                    current.boundary,
                    current.context_fingerprints,
                    current.tracker_id,
                ),
                "shape",
            ),
        )
        for invalid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_task_update(previous, invalid)


class TaskObjectiveTests(unittest.TestCase):
    @staticmethod
    def _encoding(*, requires_grad=False):
        from icgs.models.memories.task import TaskEncoding

        sentinel = torch.finfo(torch.float32).min
        event_valid = torch.tensor(
            [[True, True, False], [True, False, True]], dtype=torch.bool
        )
        r = torch.zeros(2, MethodConfig().event.width)
        alignment_logits = torch.tensor(
            [[0.2, -0.3, sentinel, 0.1], [-0.2, sentinel, 0.4, 0.0]],
            dtype=torch.float32,
        )
        event_logits = torch.tensor(
            [
                [[0.1, -0.2, 0.3], [-0.4, 0.2, -0.1], [0.0, 0.0, 0.0]],
                [[0.2, 0.1, -0.3], [0.0, 0.0, 0.0], [-0.2, 0.4, 0.1]],
            ],
            dtype=torch.float32,
        )
        if requires_grad:
            alignment_logits.requires_grad_(True)
            event_logits.requires_grad_(True)
        return TaskEncoding(r, alignment_logits, event_logits, event_valid)

    @staticmethod
    def _targets():
        return {
            "alignment_target": torch.tensor(
                [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            "alignment_valid": torch.tensor([True, True], dtype=torch.bool),
            "rho_target": torch.tensor(
                [[True, False, False], [False, False, True]], dtype=torch.bool
            ),
            "rho_valid": torch.tensor(
                [[True, True, False], [True, False, True]], dtype=torch.bool
            ),
            "nu_target": torch.tensor(
                [[False, True, False], [True, False, False]], dtype=torch.bool
            ),
            "nu_valid": torch.tensor(
                [[True, True, False], [True, False, False]], dtype=torch.bool
            ),
            "eligibility_target": torch.tensor(
                [[True, False, False], [False, False, True]], dtype=torch.bool
            ),
            "eligibility_valid": torch.tensor(
                [[True, False, False], [False, False, True]], dtype=torch.bool
            ),
        }

    @staticmethod
    def _masked_mean(values, mask):
        weight = mask.to(dtype=values.dtype)
        return (values * weight).sum() / weight.sum().clamp_min(1)

    def test_task_loss_matches_exact_weighted_soft_ce_and_bce_formula(self):
        from torch.nn import functional as F

        from icgs.algorithms.objectives.task import task_loss

        encoding = self._encoding()
        targets = self._targets()
        config = MethodConfig.from_dict(
            {
                "losses": {
                    "alignment_weight": 2.0,
                    "occurrence_weight": 3.0,
                    "relation_weight": 4.0,
                    "eligibility_weight": 5.0,
                }
            }
        )
        snapshots = {name: value.clone() for name, value in targets.items()}
        alignment_target = targets["alignment_target"].to(
            encoding.alignment_logits.dtype
        )
        alignment = -(
            alignment_target
            * F.log_softmax(encoding.alignment_logits, dim=-1)
        ).sum(dim=-1)
        event_targets = (
            targets["rho_target"],
            targets["nu_target"],
            targets["eligibility_target"],
        )
        event_valid = (
            targets["rho_valid"],
            targets["nu_valid"],
            targets["eligibility_valid"],
        )
        event_terms = []
        for channel, (target, valid) in enumerate(zip(event_targets, event_valid)):
            values = F.binary_cross_entropy_with_logits(
                encoding.event_logits[..., channel],
                target.to(dtype=encoding.event_logits.dtype),
                reduction="none",
            )
            event_terms.append(self._masked_mean(values, valid))
        expected = (
            2.0 * self._masked_mean(alignment, targets["alignment_valid"])
            + 3.0 * event_terms[0]
            + 4.0 * event_terms[1]
            + 5.0 * event_terms[2]
        )
        actual = task_loss(encoding, config=config, **targets)
        self.assertEqual(actual.ndim, 0)
        self.assertTrue(targets["rho_valid"][0, 0].item())
        self.assertTrue(targets["rho_target"][0, 0].item())
        self.assertTrue(targets["nu_valid"][0, 0].item())
        self.assertFalse(targets["nu_target"][0, 0].item())
        torch.testing.assert_close(actual, expected)
        for name, snapshot in snapshots.items():
            torch.testing.assert_close(targets[name], snapshot)

    def test_free_space_nu_mask_does_not_mask_valid_eligibility(self):
        from torch.nn import functional as F

        from icgs.algorithms.objectives.task import task_loss

        encoding = self._encoding()
        targets = self._targets()
        targets["alignment_valid"].zero_()
        targets["alignment_target"].zero_()
        targets["rho_valid"].zero_()
        targets["rho_target"].zero_()
        targets["nu_valid"].zero_()
        targets["nu_target"].zero_()
        targets["eligibility_valid"].zero_()
        targets["eligibility_target"].zero_()
        targets["eligibility_valid"][0, 0] = True
        targets["eligibility_target"][0, 0] = True
        config = MethodConfig.from_dict(
            {
                "losses": {
                    "alignment_weight": 0.0,
                    "occurrence_weight": 0.0,
                    "relation_weight": 0.0,
                    "eligibility_weight": 1.0,
                }
            }
        )
        actual = task_loss(encoding, config=config, **targets)
        expected = F.binary_cross_entropy_with_logits(
            encoding.event_logits[0, 0, 2],
            torch.ones((), dtype=encoding.event_logits.dtype),
        )
        torch.testing.assert_close(actual, expected)

    def test_auxiliary_masks_require_event_valid_and_false_placeholders(self):
        from icgs.algorithms.objectives.task import task_loss

        encoding = self._encoding()
        base = self._targets()
        cases = []
        bad_placeholder = {name: value.clone() for name, value in base.items()}
        bad_placeholder["nu_target"][0, 0] = True
        bad_placeholder["nu_valid"][0, 0] = False
        cases.append((bad_placeholder, "nu_target.*placeholder"))
        bad_padding = {name: value.clone() for name, value in base.items()}
        bad_padding["rho_valid"][0, 2] = True
        cases.append((bad_padding, "rho_valid.*event_valid"))
        for invalid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    task_loss(encoding, config=MethodConfig(), **invalid)

    def test_alignment_target_validation_and_null_last(self):
        from icgs.algorithms.objectives.task import task_loss

        encoding = self._encoding()
        base = self._targets()
        cases = []

        padding_mass = {name: value.clone() for name, value in base.items()}
        padding_mass["alignment_target"][0] = torch.tensor(
            [0.45, 0.45, 0.1, 0.0], dtype=torch.float64
        )
        cases.append((padding_mass, "invalid event"))

        masked_nonzero = {name: value.clone() for name, value in base.items()}
        masked_nonzero["alignment_valid"][0] = False
        cases.append((masked_nonzero, "zero placeholder"))

        bad_sum = {name: value.clone() for name, value in base.items()}
        bad_sum["alignment_target"][0] = torch.tensor(
            [0.4, 0.4, 0.0, 0.0], dtype=torch.float64
        )
        cases.append((bad_sum, "sum to one"))

        negative = {name: value.clone() for name, value in base.items()}
        negative["alignment_target"][0] = torch.tensor(
            [-0.1, 0.6, 0.0, 0.5], dtype=torch.float64
        )
        cases.append((negative, "nonnegative"))

        nonfinite = {name: value.clone() for name, value in base.items()}
        nonfinite["alignment_target"][0, 0] = float("nan")
        cases.append((nonfinite, "finite"))

        for invalid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    task_loss(encoding, config=MethodConfig(), **invalid)

    def test_empty_supervision_pools_are_differentiable_zero(self):
        from icgs.algorithms.objectives.task import task_loss

        encoding = self._encoding(requires_grad=True)
        targets = self._targets()
        for name in (
            "alignment_target",
            "alignment_valid",
            "rho_target",
            "rho_valid",
            "nu_target",
            "nu_valid",
            "eligibility_target",
            "eligibility_valid",
        ):
            targets[name].zero_()
        loss = task_loss(encoding, config=MethodConfig(), **targets)
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        loss.backward()
        self.assertIsNotNone(encoding.alignment_logits.grad)
        self.assertIsNotNone(encoding.event_logits.grad)
        torch.testing.assert_close(
            encoding.alignment_logits.grad,
            torch.zeros_like(encoding.alignment_logits.grad),
        )
        torch.testing.assert_close(
            encoding.event_logits.grad,
            torch.zeros_like(encoding.event_logits.grad),
        )

    def test_task_loss_gradients_cover_four_heads_and_exclude_masked_entries(self):
        from icgs.algorithms.objectives.task import task_loss

        encoding = self._encoding(requires_grad=True)
        targets = self._targets()
        targets["alignment_valid"][1] = False
        targets["alignment_target"][1].zero_()
        loss = task_loss(encoding, config=MethodConfig(), **targets)
        loss.backward()
        alignment_grad = encoding.alignment_logits.grad
        event_grad = encoding.event_logits.grad
        self.assertIsNotNone(alignment_grad)
        self.assertIsNotNone(event_grad)
        self.assertTrue(torch.isfinite(alignment_grad).all().item())
        self.assertTrue(torch.isfinite(event_grad).all().item())
        self.assertTrue((alignment_grad[targets["alignment_valid"]] != 0).any().item())
        torch.testing.assert_close(
            alignment_grad[~targets["alignment_valid"]],
            torch.zeros_like(alignment_grad[~targets["alignment_valid"]]),
        )
        torch.testing.assert_close(
            alignment_grad[:, :-1][~encoding.event_valid],
            torch.zeros_like(alignment_grad[:, :-1][~encoding.event_valid]),
        )
        for channel, name in enumerate(("rho", "nu", "eligibility")):
            valid = targets[f"{name}_valid"]
            self.assertTrue((event_grad[..., channel][valid] != 0).all().item())
            torch.testing.assert_close(
                event_grad[..., channel][~valid],
                torch.zeros_like(event_grad[..., channel][~valid]),
            )

    def test_task_loss_rejects_bad_shapes_dtypes_and_config(self):
        from icgs.algorithms.objectives.task import task_loss

        encoding = self._encoding()
        base = self._targets()
        cases = []
        bad_alignment_dtype = {name: value.clone() for name, value in base.items()}
        bad_alignment_dtype["alignment_target"] = bad_alignment_dtype[
            "alignment_target"
        ].to(torch.long)
        cases.append((bad_alignment_dtype, "alignment_target.*floating"))
        bad_target_dtype = {name: value.clone() for name, value in base.items()}
        bad_target_dtype["rho_target"] = bad_target_dtype["rho_target"].float()
        cases.append((bad_target_dtype, "rho_target.*bool"))
        bad_mask_dtype = {name: value.clone() for name, value in base.items()}
        bad_mask_dtype["eligibility_valid"] = bad_mask_dtype[
            "eligibility_valid"
        ].long()
        cases.append((bad_mask_dtype, "eligibility_valid.*bool"))
        bad_shape = {name: value.clone() for name, value in base.items()}
        bad_shape["nu_target"] = bad_shape["nu_target"][:, :-1]
        cases.append((bad_shape, "nu_target.*shape"))
        bad_device = {name: value.clone() for name, value in base.items()}
        bad_device["alignment_valid"] = torch.ones(
            2, dtype=torch.bool, device="meta"
        )
        cases.append((bad_device, "alignment_valid.*device"))
        for invalid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    task_loss(encoding, config=MethodConfig(), **invalid)
        with self.assertRaisesRegex(TypeError, "MethodConfig"):
            task_loss(encoding, config=object(), **base)

    def test_task_loss_rejects_malformed_raw_task_encoding(self):
        from icgs.algorithms.objectives.task import task_loss

        encoding = self._encoding()
        targets = self._targets()

        nonfinite_alignment = encoding.alignment_logits.clone()
        nonfinite_alignment[0, 0] = float("nan")
        nonfinite_events = encoding.event_logits.clone()
        nonfinite_events[0, 0, 0] = float("inf")
        bad_padding_alignment = encoding.alignment_logits.clone()
        bad_padding_alignment[0, 2] = 0.0
        bad_padding_events = encoding.event_logits.clone()
        bad_padding_events[0, 2, 0] = 0.1
        cases = (
            (replace(encoding, event_logits=encoding.event_logits[..., :2]), "event_logits.*shape"),
            (replace(encoding, event_valid=encoding.event_valid.long()), "event_valid.*bool"),
            (replace(encoding, alignment_logits=nonfinite_alignment), "alignment_logits.*finite"),
            (replace(encoding, event_logits=nonfinite_events), "event_logits.*finite"),
            (replace(encoding, alignment_logits=bad_padding_alignment), "sentinel"),
            (replace(encoding, event_logits=bad_padding_events), "exactly zero"),
        )
        for invalid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    task_loss(invalid, config=MethodConfig(), **targets)

    def test_task_objective_has_narrow_dependency_and_api_surface(self):
        import icgs.algorithms.objectives.task as task_module
        from icgs.algorithms.objectives.task import task_loss

        parameters = tuple(inspect.signature(task_loss).parameters)
        self.assertEqual(
            parameters,
            (
                "encoding",
                "alignment_target",
                "alignment_valid",
                "rho_target",
                "rho_valid",
                "nu_target",
                "nu_valid",
                "eligibility_target",
                "eligibility_valid",
                "config",
            ),
        )
        source = inspect.getsource(task_module)
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        self.assertLessEqual(
            imported,
            {
                "__future__",
                "torch",
                "torch.nn",
                "icgs.configuration.method",
                "icgs.models.memories.task",
            },
        )
        for forbidden in (
            "TaskState",
            "EventMemory",
            "MethodContext",
            "InstantPolicy",
            "router",
            "data.collection.annotations",
            "program_id",
            "rlbench",
        ):
            self.assertNotIn(forbidden, source)


class RouterValidationTests(unittest.TestCase):
    @staticmethod
    def _tensors():
        return (
            torch.tensor([[0.4, 0.6]], dtype=torch.float32),
            torch.ones(1, 2, dtype=torch.float32),
            torch.tensor([[True, False]], dtype=torch.bool),
        )

    def test_router_rejects_malformed_tensors_and_configuration(self):
        from icgs.algorithms.planning.router import router_probabilities

        alpha, eligible, valid = self._tensors()
        cases = (
            ((alpha[0], eligible, valid, MethodConfig()), "event_alpha.*\[B,L\]"),
            ((alpha, eligible[:, :1], valid, MethodConfig()), "shape"),
            ((alpha.long(), eligible, valid, MethodConfig()), "event_alpha.*floating"),
            ((alpha, eligible.bool(), valid, MethodConfig()), "eligible.*floating"),
            ((alpha, eligible, valid.long(), MethodConfig()), "native_window_valid.*bool"),
            ((alpha, eligible.double(), valid, MethodConfig()), "dtype"),
            ((alpha, eligible, valid, object()), "RouterConfig or MethodConfig"),
        )
        for arguments, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    router_probabilities(*arguments)

        for name, index, value in (
            ("event_alpha", (0, 0), float("nan")),
            ("eligible", (0, 0), float("inf")),
            ("event_alpha", (0, 0), -0.1),
            ("eligible", (0, 0), 1.1),
        ):
            with self.subTest(name=name, value=value):
                values = {
                    "event_alpha": alpha.clone(),
                    "eligible": eligible.clone(),
                    "native_window_valid": valid.clone(),
                }
                values[name][index] = value
                with self.assertRaisesRegex(ValueError, f"{name}.*(finite|\[0,1\])"):
                    router_probabilities(config=MethodConfig(), **values)

        with self.assertRaises(TypeError):
            router_probabilities(alpha, eligible, valid)
        with self.assertRaisesRegex(ValueError, "probability_epsilon.*strictly positive"):
            router_probabilities(
                alpha,
                eligible,
                valid,
                replace(MethodConfig().router, probability_epsilon=0.0),
            )

    def test_window_rejects_malformed_same_demo_partition(self):
        from icgs.algorithms.planning.router import select_window_indices
        from icgs.contracts.method import SegmentRef

        target = SegmentRef("demo-a", 4, 9, "interaction", True)
        malformed = (
            (
                (
                    SegmentRef("demo-a", 0, 4, "interaction", True),
                    target,
                    SegmentRef("demo-a", 10, 13, "interaction", True),
                ),
                "contiguous",
            ),
            (
                (
                    SegmentRef("demo-a", 0, 5, "interaction", True),
                    SegmentRef("demo-a", 4, 9, "interaction", True),
                ),
                "contiguous",
            ),
            ((SegmentRef("demo-a", 4, 4, "interaction", True),), "positive"),
            (
                (
                    SegmentRef("demo-a", 0, 4, "interaction", True),
                    SegmentRef("demo-b", 4, 9, "interaction", True),
                ),
                "demo_content_hash",
            ),
            ((SegmentRef("demo-a", 0, 0, "start", False), target), "interaction"),
        )
        for interactions, message in malformed:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    select_window_indices(
                        target,
                        interactions,
                        (),
                        MethodConfig(),
                        native_waypoint_count=5,
                    )

        valid_partition = (
            SegmentRef("demo-a", 0, 4, "interaction", True),
            target,
            SegmentRef("demo-a", 9, 13, "interaction", True),
        )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            select_window_indices(
                SegmentRef("demo-a", 1, 3, "interaction", True),
                valid_partition,
                (),
                MethodConfig(),
                native_waypoint_count=5,
            )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            select_window_indices(
                target,
                (target, target),
                (),
                MethodConfig(),
                native_waypoint_count=5,
            )

    def test_window_rejects_malformed_transition_indices_and_budget(self):
        from icgs.algorithms.planning.router import select_window_indices
        from icgs.contracts.method import SegmentRef

        target = SegmentRef("demo-a", 0, 9, "interaction", True)
        cases = (
            ((2, 1), 5, MethodConfig(), "sorted"),
            ((2, 2), 5, MethodConfig(), "unique"),
            ((True,), 5, MethodConfig(), "integer"),
            ((10,), 5, MethodConfig(), "demo range"),
            ((), 0, MethodConfig(), "positive integer"),
            ((), True, MethodConfig(), "positive integer"),
            ((), 5, None, "RouterConfig or MethodConfig"),
        )
        for transitions, count, config, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    select_window_indices(
                        target,
                        (target,),
                        transitions,
                        config,
                        native_waypoint_count=count,
                    )


class RouterNumericalTests(unittest.TestCase):
    def test_router_primary_mixture_and_exact_invalid_zero(self):
        from icgs.algorithms.planning.router import router_probabilities

        probabilities = router_probabilities(
            torch.tensor([[0.4, 0.6]]),
            torch.ones(1, 2),
            torch.tensor([[True, False]]),
            MethodConfig(),
        )
        torch.testing.assert_close(probabilities, torch.tensor([[0.5, 0.5, 0.0]]))
        torch.testing.assert_close(
            probabilities.sum(dim=-1), torch.ones(probabilities.shape[0])
        )

    def test_router_fallback_is_independent_per_batch_row(self):
        from icgs.algorithms.planning.router import router_probabilities

        probabilities = router_probabilities(
            torch.tensor([[0.4, 0.6], [0.2, 0.8]]),
            torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
            torch.tensor([[True, False], [True, True]]),
            MethodConfig().router,
        )
        torch.testing.assert_close(
            probabilities,
            torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]),
        )

    def test_router_fallback_threshold_comparison_is_strict(self):
        from icgs.algorithms.planning.router import router_probabilities

        config = replace(
            MethodConfig().router,
            full_context_probability=0.25,
            probability_epsilon=0.125,
            fallback_threshold=0.125,
        )
        probabilities = router_probabilities(
            torch.zeros(1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            config,
        )
        torch.testing.assert_close(probabilities, torch.tensor([[0.25, 0.75]]))

    def test_router_preserves_fp16_subnormal_nonfallback_normalization(self):
        from icgs.algorithms.planning.router import router_probabilities

        config = replace(
            MethodConfig().router,
            probability_epsilon=1e-6,
            fallback_threshold=1e-6,
        )
        probabilities = router_probabilities(
            torch.zeros(1, 1, dtype=torch.float16),
            torch.ones(1, 1, dtype=torch.float16),
            torch.ones(1, 1, dtype=torch.bool),
            config,
        )
        torch.testing.assert_close(
            probabilities,
            torch.tensor([[0.5, 0.5]], dtype=torch.float16),
        )
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones(1, dtype=torch.float16),
        )

    def test_router_invalid_windows_have_zero_probability_and_gradient(self):
        from icgs.algorithms.planning.router import router_probabilities

        event_alpha = torch.tensor([[0.2, 0.3, 0.5]], requires_grad=True)
        eligible = torch.tensor([[0.7, 0.8, 0.9]], requires_grad=True)
        valid = torch.tensor([[True, False, True]])
        probabilities = router_probabilities(
            event_alpha, eligible, valid, MethodConfig()
        )
        self.assertEqual(probabilities[0, 2].item(), 0.0)
        probabilities[..., 1:].sum().backward()
        torch.testing.assert_close(
            event_alpha.grad[~valid], torch.zeros_like(event_alpha.grad[~valid])
        )
        torch.testing.assert_close(
            eligible.grad[~valid], torch.zeros_like(eligible.grad[~valid])
        )


class RouteRngProtocolTests(unittest.TestCase):
    @staticmethod
    def _assert_numpy_state_equal(left, right):
        if left[0] != right[0] or left[2:] != right[2:]:
            raise AssertionError("NumPy global RNG metadata changed")
        np.testing.assert_array_equal(left[1], right[1])

    def test_split_route_seed_matches_exact_seed_sequence_children(self):
        from icgs.algorithms.planning.router import split_route_seed

        self.assertIn("route and diffusion seeds", inspect.getdoc(split_route_seed))
        expected = tuple(
            int(child.generate_state(1)[0])
            for child in np.random.SeedSequence(17).spawn(2)
        )
        self.assertEqual(expected, (3302413169, 2035845825))
        self.assertEqual(split_route_seed(17), expected)
        self.assertEqual(split_route_seed(17), split_route_seed(17))
        self.assertNotEqual(*split_route_seed(17))

    def test_route_draw_is_deterministic_and_never_selects_zero_mass(self):
        from icgs.algorithms.planning.router import draw_route

        self.assertEqual(draw_route(torch.tensor([0.0, 1.0]), route_seed=7), 1)
        draws = tuple(
            draw_route(torch.tensor([0.5, 0.0, 0.5]), route_seed=seed)
            for seed in range(64)
        )
        self.assertTrue(set(draws).issubset({0, 2}))
        self.assertEqual(
            draw_route(torch.tensor([0.25, 0.75]), route_seed=29),
            draw_route(torch.tensor([0.25, 0.75]), route_seed=29),
        )

        boundary_draw = np.random.Generator(np.random.PCG64(123)).random()
        boundary_probabilities = torch.tensor(
            [boundary_draw, 0.0, 1.0 - boundary_draw], dtype=torch.float64
        )
        self.assertEqual(draw_route(boundary_probabilities, route_seed=123), 2)

    def test_route_draw_canonicalizes_accepted_near_unit_mass(self):
        from icgs.algorithms.planning.router import draw_route

        cases = (
            (torch.tensor([0.5, 0.4999982, 0.0]), 339728),
            (torch.tensor([0.5, 0.5000005, 0.0]), 41),
        )
        for probabilities, route_seed in cases:
            with self.subTest(probabilities=probabilities, route_seed=route_seed):
                values = probabilities.to(dtype=torch.float64).numpy().copy()
                values /= values.sum(dtype=np.float64)
                cumulative = np.cumsum(values, dtype=np.float64)
                cumulative[-1] = 1.0
                draw = np.random.Generator(np.random.PCG64(route_seed)).random()
                expected = int(np.searchsorted(cumulative, draw, side="right"))

                actual = draw_route(probabilities, route_seed=route_seed)
                self.assertEqual(actual, expected)
                self.assertNotEqual(actual, 2)

    def test_route_draw_rejects_distribution_outside_tolerance(self):
        from icgs.algorithms.planning.router import draw_route

        cases = (
            (torch.tensor(1.0), "one-dimensional"),
            (torch.ones(1, 2), "one-dimensional"),
            (torch.empty(0), "nonempty"),
            (torch.tensor([0, 1]), "floating"),
            (torch.tensor([float("nan"), 1.0]), "finite"),
            (torch.tensor([float("inf"), 0.0]), "finite"),
            (torch.tensor([-0.1, 1.1]), "nonnegative"),
            (torch.tensor([0.4, 0.5]), "sum to one"),
            (torch.tensor([0.6, 0.6]), "sum to one"),
        )
        for probabilities, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    draw_route(probabilities, route_seed=11)

    def test_route_seed_validation_rejects_bool_negative_and_noninteger(self):
        from icgs.algorithms.planning.router import draw_route, split_route_seed

        for invalid in (True, -1, 1.5, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex((TypeError, ValueError), "nonnegative integer"):
                    split_route_seed(invalid)
                with self.assertRaisesRegex((TypeError, ValueError), "nonnegative integer"):
                    draw_route(torch.tensor([1.0]), route_seed=invalid)

    def test_route_rng_success_and_failure_preserve_global_rng_and_input(self):
        from icgs.algorithms.planning.router import draw_route, split_route_seed

        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        probabilities = torch.tensor([0.2, 0.3, 0.5])
        probabilities_before = probabilities.clone()

        split_route_seed(31)
        draw_route(probabilities, route_seed=37)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            draw_route(torch.tensor([0.2, 0.2]), route_seed=41)

        self.assertEqual(random.getstate(), python_before)
        self._assert_numpy_state_equal(np.random.get_state(), numpy_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        torch.testing.assert_close(probabilities, probabilities_before)

    def test_route_rng_protocol_and_dependency_boundary(self):
        import icgs.algorithms.planning.router as router_module
        from icgs.algorithms.planning.router import (
            ROUTE_RNG_PROTOCOL,
            draw_route,
            split_route_seed,
        )

        self.assertEqual(ROUTE_RNG_PROTOCOL, "pcg64-v1")
        self.assertEqual(tuple(inspect.signature(split_route_seed).parameters), ("seed",))
        self.assertEqual(
            tuple(inspect.signature(draw_route).parameters),
            ("probabilities", "route_seed"),
        )
        self.assertEqual(
            inspect.signature(draw_route).parameters["route_seed"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        source = inspect.getsource(router_module)
        for forbidden in (
            "PreparedContext",
            "InstantPolicy",
            "MethodContext",
            "Candidate",
            "composition",
            "state.randomness",
        ):
            self.assertNotIn(forbidden, source)


class StructuralWindowTests(unittest.TestCase):
    @staticmethod
    def _ref(a, b, *, valid=True):
        from icgs.contracts.method import SegmentRef

        return SegmentRef("demo-a", a, b, "interaction", valid)

    def test_shared_endpoint_partition_selects_exact_window(self):
        from icgs.algorithms.planning.router import select_window_indices

        interactions = (self._ref(0, 4), self._ref(4, 9), self._ref(9, 13))
        actual = select_window_indices(
            interactions[1],
            interactions,
            (6,),
            MethodConfig(),
            native_waypoint_count=10,
        )
        self.assertEqual(actual, (0, 1, 3, 4, 6, 7, 9, 10, 12, 13))

    def test_window_consumes_p05_debounced_confirmation_indices(self):
        from icgs.algorithms.planning.router import select_window_indices
        from icgs.data.preprocessing.events import debounced_grip_boundaries

        config = MethodConfig()
        transitions = debounced_grip_boundaries(
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1], config.event
        )
        self.assertEqual(transitions, (4,))
        target = self._ref(0, 9)
        self.assertEqual(
            select_window_indices(
                target,
                (target,),
                transitions,
                config.router,
                native_waypoint_count=5,
            ),
            (0, 1, 4, 8, 9),
        )

    def test_window_neighbors_are_bounded_at_partition_edges(self):
        from icgs.algorithms.planning.router import select_window_indices

        interactions = (self._ref(0, 4), self._ref(4, 9), self._ref(9, 13))
        first = select_window_indices(
            interactions[0], interactions, (), MethodConfig(), native_waypoint_count=6
        )
        last = select_window_indices(
            interactions[-1], interactions, (), MethodConfig(), native_waypoint_count=6
        )
        self.assertEqual(first, (0, 1, 4, 5, 8, 9))
        self.assertEqual(last, (4, 5, 8, 9, 12, 13))

    def test_uniform_fill_rank_examples_lock_earlier_ties(self):
        from icgs.algorithms.planning.router import select_window_indices

        single = (self._ref(1, 9),)
        self.assertEqual(
            select_window_indices(
                single[0], single, (3, 5, 7), MethodConfig(), native_waypoint_count=6
            ),
            (1, 3, 4, 5, 7, 9),
        )
        self.assertEqual(
            select_window_indices(
                single[0], single, (3, 5, 7), MethodConfig(), native_waypoint_count=8
            ),
            (1, 2, 3, 4, 5, 7, 8, 9),
        )

        wider = (self._ref(0, 14),)
        self.assertEqual(
            select_window_indices(
                wider[0],
                wider,
                (2, 4, 6, 8, 10, 12),
                MethodConfig(),
                native_waypoint_count=12,
            ),
            (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14),
        )

    def test_valid_but_impossible_windows_return_none(self):
        from icgs.algorithms.planning.router import select_window_indices

        target = self._ref(0, 9)
        self.assertIsNone(
            select_window_indices(
                target,
                (target,),
                (1, 2, 3, 4, 5),
                MethodConfig(),
                native_waypoint_count=4,
            )
        )
        self.assertIsNone(
            select_window_indices(
                target,
                (target,),
                (),
                MethodConfig(),
                native_waypoint_count=11,
            )
        )

    def test_window_target_must_be_structurally_eligible(self):
        from icgs.algorithms.planning.router import select_window_indices

        target = self._ref(0, 9, valid=False)
        with self.assertRaisesRegex(ValueError, "structurally eligible"):
            select_window_indices(
                target, (target,), (), MethodConfig(), native_waypoint_count=5
            )

    def test_router_and_window_selection_do_not_mutate_inputs(self):
        from icgs.algorithms.planning.router import (
            router_probabilities,
            select_window_indices,
        )

        event_alpha = torch.tensor([[0.4, 0.6]])
        eligible = torch.tensor([[0.7, 0.8]])
        native_window_valid = torch.tensor([[True, False]])
        original_tensors = tuple(
            value.clone() for value in (event_alpha, eligible, native_window_valid)
        )
        router_probabilities(
            event_alpha, eligible, native_window_valid, MethodConfig()
        )
        for actual, original in zip(
            (event_alpha, eligible, native_window_valid), original_tensors
        ):
            torch.testing.assert_close(actual, original)

        target = self._ref(0, 9)
        interactions = (target,)
        transitions = [4]
        original_interactions = tuple(interactions)
        original_transitions = list(transitions)
        select_window_indices(
            target,
            interactions,
            transitions,
            MethodConfig(),
            native_waypoint_count=5,
        )
        self.assertEqual(interactions, original_interactions)
        self.assertEqual(transitions, original_transitions)

    def test_router_module_has_narrow_dependency_and_api_surface(self):
        import icgs.algorithms.planning.router as router_module
        from icgs.algorithms.planning.router import (
            router_probabilities,
            select_window_indices,
        )

        self.assertEqual(
            tuple(inspect.signature(router_probabilities).parameters),
            ("event_alpha", "eligible", "native_window_valid", "config"),
        )
        self.assertEqual(
            tuple(inspect.signature(select_window_indices).parameters),
            (
                "target",
                "demo_interactions",
                "grip_transition_indices",
                "config",
                "native_waypoint_count",
            ),
        )
        self.assertEqual(
            inspect.signature(select_window_indices).parameters[
                "native_waypoint_count"
            ].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        tree = ast.parse(inspect.getsource(router_module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        self.assertLessEqual(
            imported,
            {
                "__future__",
                "collections.abc",
                "math",
                "numbers",
                "numpy",
                "torch",
                "icgs.configuration.method",
                "icgs.contracts.method",
            },
        )


if __name__ == "__main__":
    unittest.main()
