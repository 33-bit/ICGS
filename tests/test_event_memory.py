import math
import inspect
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import torch

from icgs.configuration.method import MethodConfig
from icgs.contracts.method import ExecutedTransition, TimedCommand, TimedObservation
from icgs.contracts.records import Observation


class EventSegmentationTests(unittest.TestCase):
    @staticmethod
    def _pose(x=0.0, rotation_deg=0.0):
        angle = math.radians(rotation_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        pose = np.eye(4)
        pose[:3, :3] = np.array([
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ])
        pose[0, 3] = x
        return pose

    @classmethod
    def _transitions(cls, xs, *, rotations=None, grips=None, commands=None, boundaries=None):
        count = len(xs)
        rotations = [0.0] * count if rotations is None else rotations
        grips = [0] * count if grips is None else grips
        boundaries = list(range(count)) if boundaries is None else boundaries
        if not (len(rotations) == len(grips) == len(boundaries) == count):
            raise AssertionError("fixture fields must have equal lengths")
        observations = tuple(
            TimedObservation(
                Observation(np.zeros((1, 3)), cls._pose(x, angle), grip),
                boundary,
                float(index),
                float(index),
                "sensor-v1",
            )
            for index, (x, angle, grip, boundary) in enumerate(
                zip(xs, rotations, grips, boundaries)
            )
        )
        if commands is None:
            commands = tuple(TimedCommand(np.eye(4), 0, 0.1) for _ in range(count - 1))
        if len(commands) != count - 1:
            raise AssertionError("fixture requires one command per transition")
        return tuple(
            ExecutedTransition(
                observations[index], observations[index + 1], commands[index], 0.1, 1, "ok"
            )
            for index in range(count - 1)
        )

    @staticmethod
    def _positions(final_boundary, jump_boundaries):
        position = 0.0
        result = []
        for boundary in range(final_boundary + 1):
            if boundary in jump_boundaries:
                position += 0.13
            result.append(position)
        return result

    @staticmethod
    def _structure(segments):
        return tuple(
            (segment.a, segment.b, segment.kind, segment.valid_action_window)
            for segment in segments
        )

    @staticmethod
    def _interactions(segments):
        return tuple((segment.a, segment.b) for segment in segments if segment.kind == "interaction")

    @staticmethod
    def _config(**changes):
        return replace(MethodConfig().event, **changes)

    def test_timed_demo_input_validates_without_transforming(self):
        from icgs.data.preprocessing.events import TimedDemoInput

        transitions = self._transitions([0.0, 0.0])
        demo = TimedDemoInput(transitions, " hash-with-owner-spacing ")
        self.assertIs(demo.transitions, transitions)
        self.assertEqual(demo.demo_content_hash, " hash-with-owner-spacing ")

        with self.assertRaisesRegex(TypeError, "tuple"):
            TimedDemoInput(list(transitions), "hash")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            TimedDemoInput((), "hash")
        with self.assertRaisesRegex(TypeError, "ExecutedTransition"):
            TimedDemoInput((object(),), "hash")
        with self.assertRaisesRegex(TypeError, "string"):
            TimedDemoInput(transitions, 1)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            TimedDemoInput(transitions, "  ")

        nonadjacent = self._transitions([0.0, 0.0], boundaries=[0, 2])
        with self.assertRaisesRegex(ValueError, "adjacent"):
            TimedDemoInput(nonadjacent, "hash")

        first = self._transitions([0.0, 0.0], boundaries=[0, 1])[0]
        second = self._transitions([0.0, 0.0], boundaries=[2, 3])[0]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            TimedDemoInput((first, second), "hash")

    def test_debounced_grip_boundaries_confirm_without_backdating(self):
        from icgs.data.preprocessing.events import debounced_grip_boundaries

        config = self._config()
        self.assertEqual(debounced_grip_boundaries([0, 1, 0, 1, 1], config), (4,))
        self.assertEqual(debounced_grip_boundaries([1, 1, 0, 0], config), (3,))
        self.assertEqual(debounced_grip_boundaries([0], config), ())
        self.assertEqual(debounced_grip_boundaries([0, 0, 0], config), ())
        with self.assertRaisesRegex(ValueError, "non-empty"):
            debounced_grip_boundaries([], config)
        with self.assertRaisesRegex(ValueError, "grip"):
            debounced_grip_boundaries([0, 2], config)
        with self.assertRaisesRegex(TypeError, "config"):
            debounced_grip_boundaries([0], None)

    def test_segment_demo_uses_measured_state_not_commands_and_preserves_hash(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        class GuardedCommand(TimedCommand):
            _guarded = False

            def __getattribute__(self, name):
                if name in {"target_w", "grip"} and object.__getattribute__(self, "_guarded"):
                    raise AssertionError("segment_demo read a command field")
                return super().__getattribute__(name)

        guarded = GuardedCommand(self._pose(9.0, 170.0), 1, 0.1)
        object.__setattr__(guarded, "_guarded", True)
        ordinary = TimedCommand(self._pose(-9.0, -170.0), 0, 0.1)
        transitions_a = self._transitions([0.0, 0.0], commands=(guarded,))
        transitions_b = self._transitions([0.0, 0.0], commands=(ordinary,))

        segments_a = segment_demo(TimedDemoInput(transitions_a, "hash-a"), self._config())
        segments_b = segment_demo(TimedDemoInput(transitions_b, "hash-b"), self._config())
        self.assertEqual(self._structure(segments_a), self._structure(segments_b))
        self.assertEqual(
            self._structure(segments_a),
            ((0, 0, "start", False), (0, 1, "interaction", True), (1, 1, "end", False)),
        )
        self.assertTrue(all(segment.demo_content_hash == "hash-a" for segment in segments_a))
        self.assertTrue(all(segment.demo_content_hash == "hash-b" for segment in segments_b))
        with self.assertRaisesRegex(TypeError, "config"):
            segment_demo(TimedDemoInput(transitions_b, "hash-b"), None)

    def test_cumulative_translation_and_rotation_use_strict_motion_thresholds(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=1, max_intervals=100)
        exact_translation = TimedDemoInput(self._transitions([0.0, 0.06, 0.12, 0.12]), "exact-t")
        over_translation = TimedDemoInput(self._transitions([0.0, 0.061, 0.122, 0.122]), "over-t")
        self.assertEqual(self._interactions(segment_demo(exact_translation, config)), ((0, 3),))
        self.assertEqual(self._interactions(segment_demo(over_translation, config)), ((0, 2), (2, 3)))

        exact_rotation = TimedDemoInput(
            self._transitions([0.0] * 4, rotations=[0.0, 15.0, 30.0, 30.0]), "exact-r"
        )
        over_rotation = TimedDemoInput(
            self._transitions([0.0] * 4, rotations=[0.0, 15.1, 30.2, 30.2]), "over-r"
        )
        self.assertEqual(self._interactions(segment_demo(exact_rotation, config)), ((0, 3),))
        self.assertEqual(self._interactions(segment_demo(over_rotation, config)), ((0, 2), (2, 3)))

    def test_time_boundary_counts_transitions_not_observations(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=1, max_intervals=20)
        demo = TimedDemoInput(self._transitions([0.0] * 22), "time")
        self.assertEqual(self._interactions(segment_demo(demo, config)), ((0, 20), (20, 21)))

    def test_short_segments_merge_with_shorter_duration_neighbor_then_earlier(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=3, max_intervals=100)
        cases = (
            (16, {5, 7}, ((0, 7), (7, 16))),
            (16, {9, 11}, ((0, 9), (9, 16))),
            (12, {5, 7}, ((0, 7), (7, 12))),
        )
        for final_boundary, jumps, expected in cases:
            with self.subTest(final_boundary=final_boundary, jumps=jumps):
                demo = TimedDemoInput(
                    self._transitions(self._positions(final_boundary, jumps)), "short"
                )
                self.assertEqual(self._interactions(segment_demo(demo, config)), expected)

    def test_short_merge_never_erases_a_protected_grip_boundary(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=3, max_intervals=100)
        grips = [0] * 4 + [1] * 12
        demo = TimedDemoInput(
            self._transitions(self._positions(15, {7}), grips=grips), "protected-short"
        )
        self.assertEqual(self._interactions(segment_demo(demo, config)), ((0, 5), (5, 15)))

    def test_capacity_merge_uses_minimum_combined_duration_then_earlier_pair(self):
        from icgs.data.preprocessing.events import TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=1, max_intervals=100, max_interactions=2)
        demo = TimedDemoInput(self._transitions(self._positions(20, {4, 10, 13})), "cap")
        self.assertEqual(self._interactions(segment_demo(demo, config)), ((0, 13), (13, 20)))

        tied = TimedDemoInput(self._transitions(self._positions(12, {4, 8})), "cap-tie")
        self.assertEqual(self._interactions(segment_demo(tied, config)), ((0, 8), (8, 12)))

    def test_capacity_overflow_is_explicit_when_grip_boundaries_are_protected(self):
        from icgs.data.preprocessing.events import EventOverflowError, TimedDemoInput, segment_demo

        config = self._config(min_segment_intervals=1, max_intervals=100, max_interactions=2)
        grips = [0, 1, 1, 0, 0, 1, 1]
        demo = TimedDemoInput(self._transitions([0.0] * len(grips), grips=grips), "overflow")
        with self.assertRaisesRegex(EventOverflowError, "protected|overflow"):
            segment_demo(demo, config)


class EventEncoderContractTests(unittest.TestCase):
    @staticmethod
    def _inputs():
        config = MethodConfig()
        width = config.geometry.width
        frame_tokens = torch.arange(4 * 2 * width, dtype=torch.float32).reshape(
            1, 4, 2, width
        ) / 1000.0
        proprio = torch.arange(4 * 13, dtype=torch.float32).reshape(1, 4, 13) / 100.0
        return {
            "frame_tokens": frame_tokens,
            "anchor_valid": torch.ones(1, 4, 2, dtype=torch.bool),
            "proprio": proprio,
            "segment_start": torch.tensor([[0, 0, 3, 0]], dtype=torch.long),
            "segment_end": torch.tensor([[0, 3, 3, 0]], dtype=torch.long),
            "twist": torch.tensor(
                [[[0.0] * 6, [0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6, [0.0] * 6]],
                dtype=torch.float32,
            ),
            "grip_start": torch.tensor([[[0.0], [0.0], [1.0], [0.0]]]),
            "grip_end": torch.tensor([[[0.0], [1.0], [1.0], [0.0]]]),
            "event_kind": torch.tensor([[0, 1, 2, -1]], dtype=torch.long),
            "event_valid": torch.tensor([[True, True, True, False]], dtype=torch.bool),
            "local_index": torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
            "local_count": torch.tensor([[3, 3, 3, 0]], dtype=torch.long),
        }

    @staticmethod
    def _clone(inputs):
        return {name: value.clone() for name, value in inputs.items()}

    def test_event_features_primary_width_and_strict_nonfinite_contract(self):
        from icgs.models.encoders.event import event_features

        width = MethodConfig().event.width
        values = [
            torch.zeros(1, width),
            torch.zeros(1, width),
            torch.zeros(1, width),
            torch.zeros(1, 6),
            torch.zeros(1, 1),
            torch.ones(1, 1),
        ]
        self.assertEqual(tuple(event_features(*values).shape), (1, 776))

        for index in range(len(values)):
            for nonfinite in (float("nan"), float("inf")):
                with self.subTest(argument=index, nonfinite=nonfinite):
                    invalid = [value.clone() for value in values]
                    invalid[index].reshape(-1)[0] = nonfinite
                    with self.assertRaisesRegex(ValueError, "finite"):
                        event_features(*invalid)

    def test_tensor_only_surface_and_explicit_configuration(self):
        import icgs.models.encoders.event as event_module
        from icgs.models.encoders.event import EventEncoder

        self.assertEqual(
            (event_module.START_KIND, event_module.INTERACTION_KIND, event_module.END_KIND),
            (0, 1, 2),
        )
        expected = (
            "frame_tokens", "anchor_valid", "proprio", "segment_start", "segment_end",
            "twist", "grip_start", "grip_end", "event_kind", "event_valid",
            "local_index", "local_count",
        )
        parameters = tuple(inspect.signature(EventEncoder.forward).parameters)[1:]
        self.assertEqual(parameters, expected)

        source = inspect.getsource(event_module)
        for forbidden in ("SegmentRef", "EventMemory", "MethodContext"):
            self.assertNotIn(forbidden, source)

        config = MethodConfig()
        with self.assertRaisesRegex((TypeError, ValueError), "config|section"):
            EventEncoder()
        EventEncoder(method_config=config)
        section_encoder = EventEncoder(
            geometry_config=config.geometry,
            event_config=config.event,
            neural_config=config.neural,
        )
        self.assertEqual(len(section_encoder.blocks), config.event.transformer_layers)
        self.assertEqual(
            section_encoder.frame_mlp[0].in_features,
            config.geometry.width + 13,
        )
        self.assertEqual(
            section_encoder.event_mlp[0].in_features,
            3 * config.event.width + 8,
        )
        self.assertEqual(
            section_encoder.event_mlp[0].out_features,
            config.event.token_hidden_dim,
        )
        self.assertEqual(section_encoder.order_mlp[0].in_features, 2)
        self.assertEqual(
            section_encoder.blocks[0].norm_attention.eps,
            config.neural.layer_norm_eps,
        )
        with self.assertRaisesRegex(ValueError, "combine|both|section"):
            EventEncoder(
                method_config=config,
                geometry_config=config.geometry,
                event_config=config.event,
                neural_config=config.neural,
            )

    def test_encoder_returns_masked_event_encoding(self):
        from icgs.models.encoders.event import EventEncoder, EventEncoding

        config = MethodConfig()
        encoder = EventEncoder(method_config=config).eval()
        with torch.no_grad():
            encoded = encoder(**self._inputs())
        self.assertIsInstance(encoded, EventEncoding)
        self.assertEqual(tuple(encoded.tokens.shape), (1, 4, config.event.width))
        self.assertEqual(encoded.valid.dtype, torch.bool)
        torch.testing.assert_close(encoded.valid, self._inputs()["event_valid"])
        self.assertTrue(torch.isfinite(encoded.tokens).all().item())
        torch.testing.assert_close(encoded.tokens[:, 3], torch.zeros_like(encoded.tokens[:, 3]))

        gradient_inputs = self._inputs()
        for name in ("frame_tokens", "proprio", "twist", "grip_start", "grip_end"):
            gradient_inputs[name].requires_grad_(True)
        differentiable = encoder(**gradient_inputs)
        differentiable.tokens[differentiable.valid].sum().backward()
        for name in ("frame_tokens", "proprio", "twist", "grip_start", "grip_end"):
            gradient = gradient_inputs[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all().item(), name)
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in encoder.parameters()
            )
        )

    def test_generic_attention_masks_without_geometry_coordinates(self):
        from icgs.models.layers.method import MaskedSelfAttentionBlock

        config = MethodConfig()
        block = MaskedSelfAttentionBlock(
            width=config.event.width,
            neural_config=config.neural,
        ).eval()
        values = torch.randn(1, 3, config.event.width)
        valid = torch.tensor([[True, True, False]], dtype=torch.bool)
        clean = values.clone()
        clean[:, 2] = 0.0
        poisoned = clean.clone()
        poisoned[:, 2] = float("nan")
        with torch.no_grad():
            expected = block(clean, valid)
            actual = block(poisoned, valid)
        torch.testing.assert_close(actual[:, :2], expected[:, :2])
        torch.testing.assert_close(actual[:, 2], torch.zeros_like(actual[:, 2]))
        self.assertTrue(torch.isfinite(actual).all().item())

        invalid = clean.clone()
        invalid[:, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            block(invalid, valid)
        with self.assertRaisesRegex(ValueError, "valid"):
            block(clean, torch.zeros_like(valid))

    def test_dtype_and_valid_row_index_contracts(self):
        from icgs.models.encoders.event import EventEncoder

        encoder = EventEncoder(method_config=MethodConfig()).eval()
        for name, dtype in (
            ("segment_start", torch.int32),
            ("segment_end", torch.int32),
            ("anchor_valid", torch.int32),
            ("event_valid", torch.int32),
        ):
            with self.subTest(name=name):
                invalid = self._inputs()
                invalid[name] = invalid[name].to(dtype=dtype)
                with self.assertRaisesRegex(TypeError, name):
                    encoder(**invalid)

        cases = (
            ("segment_start", (0, 1), -1, "segment_start"),
            ("segment_end", (0, 1), 4, "segment_end"),
            ("local_count", (0, 1), 0, "local_count"),
            ("local_index", (0, 1), -1, "local_index"),
            ("local_index", (0, 1), 3, "local_index"),
            ("event_kind", (0, 1), 99, "event_kind"),
        )
        for name, index, value, message in cases:
            with self.subTest(name=name, value=value):
                invalid = self._inputs()
                invalid[name][index] = value
                with self.assertRaisesRegex(ValueError, message):
                    encoder(**invalid)

        reversed_segment = self._inputs()
        reversed_segment["segment_start"][0, 1] = 2
        reversed_segment["segment_end"][0, 1] = 1
        with self.assertRaisesRegex(ValueError, "start.*end|segment"):
            encoder(**reversed_segment)

    def test_valid_nonfinite_rejects_but_all_masked_padding_is_sanitized(self):
        from icgs.models.encoders.event import EventEncoder

        encoder = EventEncoder(method_config=MethodConfig()).eval()
        valid_cases = (
            ("frame_tokens", (0, 0, 0, 0), float("nan")),
            ("proprio", (0, 0, 0), float("inf")),
            ("twist", (0, 1, 0), float("nan")),
            ("grip_start", (0, 1, 0), float("inf")),
            ("grip_end", (0, 1, 0), float("nan")),
        )
        for name, index, value in valid_cases:
            with self.subTest(name=name):
                invalid = self._inputs()
                invalid[name][index] = value
                with self.assertRaisesRegex(ValueError, f"{name}.*finite|finite.*{name}"):
                    encoder(**invalid)

        clean = self._inputs()
        clean["anchor_valid"][0, 1, 1] = False
        clean["frame_tokens"][0, 1, 1] = 0.0
        poisoned = self._clone(clean)
        poisoned["frame_tokens"][0, 1, 1] = float("nan")
        poisoned["segment_start"][0, 3] = -1
        poisoned["segment_end"][0, 3] = 10_000
        poisoned["event_kind"][0, 3] = -99
        poisoned["local_index"][0, 3] = -7
        poisoned["local_count"][0, 3] = 0
        poisoned["twist"][0, 3] = float("nan")
        poisoned["grip_start"][0, 3] = float("inf")
        poisoned["grip_end"][0, 3] = -float("inf")

        with torch.no_grad():
            expected = encoder(**clean)
            actual = encoder(**poisoned)
        torch.testing.assert_close(actual.tokens[:, :3], expected.tokens[:, :3])
        torch.testing.assert_close(actual.tokens[:, 3], torch.zeros_like(actual.tokens[:, 3]))
        self.assertTrue(torch.isfinite(actual.tokens).all().item())

    def test_landmarks_are_validated_and_accepted_twist_is_not_rewritten(self):
        import icgs.models.encoders.event as event_module
        from icgs.models.encoders.event import EventEncoder

        encoder = EventEncoder(method_config=MethodConfig()).eval()
        nonzero = self._inputs()
        nonzero["twist"][0, 0, 0] = 2e-6
        with self.assertRaisesRegex(ValueError, "landmark.*twist|twist.*landmark"):
            encoder(**nonzero)

        nonpoint = self._inputs()
        nonpoint["segment_end"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "landmark.*start|landmark.*end|start.*end"):
            encoder(**nonpoint)

        accepted = self._inputs()
        accepted["twist"][0, 0, 0] = 5e-7
        original = event_module.event_features
        seen = {}

        def capture(*args, **kwargs):
            seen["twist"] = (args[3] if len(args) > 3 else kwargs["xi"]).detach().clone()
            return original(*args, **kwargs)

        with mock.patch.object(event_module, "event_features", side_effect=capture):
            with torch.no_grad():
                encoder(**accepted)
        torch.testing.assert_close(
            seen["twist"][0, 0], accepted["twist"][0, 0], rtol=0.0, atol=0.0
        )

    def test_inclusive_segment_mean_and_local_order_are_exact(self):
        import icgs.models.encoders.event as event_module
        from icgs.models.encoders.event import EventEncoder

        encoder = EventEncoder(method_config=MethodConfig()).eval()
        inputs = self._inputs()
        captured = {}
        original = event_module.event_features

        def capture_features(*args, **kwargs):
            captured["features"] = tuple(value.detach().clone() for value in args)
            return original(*args, **kwargs)

        def capture_order(_module, args):
            captured["order"] = args[0].detach().clone()

        handle = encoder.order_mlp[0].register_forward_pre_hook(capture_order)
        try:
            with mock.patch.object(event_module, "event_features", side_effect=capture_features):
                with torch.no_grad():
                    encoder(**inputs)
        finally:
            handle.remove()

        anchor_mask = inputs["anchor_valid"].unsqueeze(-1)
        pooled = torch.where(
            anchor_mask, inputs["frame_tokens"], torch.zeros_like(inputs["frame_tokens"])
        ).sum(dim=2) / anchor_mask.sum(dim=2).to(inputs["frame_tokens"].dtype)
        with torch.no_grad():
            descriptors = encoder.frame_mlp(torch.cat((pooled, inputs["proprio"]), dim=-1))
        d_start, d_end, d_mean = captured["features"][:3]
        torch.testing.assert_close(d_start[0, :3], descriptors[0, [0, 0, 3]])
        torch.testing.assert_close(d_end[0, :3], descriptors[0, [0, 3, 3]])
        torch.testing.assert_close(d_mean[0, 0], descriptors[0, 0])
        torch.testing.assert_close(d_mean[0, 1], descriptors[0, 0:4].mean(dim=0))
        torch.testing.assert_close(d_mean[0, 2], descriptors[0, 3])
        expected_order = torch.tensor(
            [[[0.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 1.0 / 3.0]]]
        )
        torch.testing.assert_close(captured["order"][:, :3], expected_order)

    def test_demo_block_permutation_is_equivariant_and_pool_invariant(self):
        from icgs.models.encoders.event import EventEncoder

        config = MethodConfig()
        width = config.geometry.width
        inputs = {
            "frame_tokens": torch.arange(6 * 2 * width, dtype=torch.float32).reshape(
                1, 6, 2, width
            ) / 1000.0,
            "anchor_valid": torch.ones(1, 6, 2, dtype=torch.bool),
            "proprio": torch.arange(6 * 13, dtype=torch.float32).reshape(1, 6, 13) / 100.0,
            "segment_start": torch.tensor([[0, 0, 2, 3, 3, 5]], dtype=torch.long),
            "segment_end": torch.tensor([[0, 2, 2, 3, 5, 5]], dtype=torch.long),
            "twist": torch.tensor(
                [[[0.0] * 6, [0.1] + [0.0] * 5, [0.0] * 6,
                  [0.0] * 6, [0.2] + [0.0] * 5, [0.0] * 6]],
                dtype=torch.float32,
            ),
            "grip_start": torch.tensor([[[0.0], [0.0], [1.0], [1.0], [1.0], [0.0]]]),
            "grip_end": torch.tensor([[[0.0], [1.0], [1.0], [1.0], [0.0], [0.0]]]),
            "event_kind": torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
            "event_valid": torch.ones(1, 6, dtype=torch.bool),
            "local_index": torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
            "local_count": torch.full((1, 6), 3, dtype=torch.long),
        }
        event_permutation = torch.tensor([3, 4, 5, 0, 1, 2])
        swapped = self._clone(inputs)
        for name in ("frame_tokens", "anchor_valid", "proprio"):
            swapped[name] = inputs[name][:, event_permutation]
        for name in (
            "twist", "grip_start", "grip_end", "event_kind", "event_valid",
            "local_index", "local_count",
        ):
            swapped[name] = inputs[name][:, event_permutation]
        # Each three-frame demo block has the same local boundary layout after swapping.
        swapped["segment_start"] = inputs["segment_start"].clone()
        swapped["segment_end"] = inputs["segment_end"].clone()

        torch.manual_seed(7)
        encoder = EventEncoder(method_config=config).eval()
        with torch.no_grad():
            original = encoder(**inputs)
            permuted = encoder(**swapped)
        torch.testing.assert_close(
            permuted.tokens, original.tokens[:, event_permutation], rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            permuted.tokens.mean(dim=1), original.tokens.mean(dim=1), rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
