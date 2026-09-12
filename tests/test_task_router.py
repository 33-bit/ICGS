import inspect
import unittest
from dataclasses import replace

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


if __name__ == "__main__":
    unittest.main()
