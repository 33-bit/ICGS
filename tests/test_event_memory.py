import hashlib
import json
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


class EventMemoryOwnershipTests(unittest.TestCase):
    @staticmethod
    def _memory_inputs(batch=1, *, requires_grad=False):
        from icgs.contracts.method import SegmentRef
        from icgs.models.encoders.event import EventEncoding

        valid = torch.tensor([[True, True, True, False]] * batch, dtype=torch.bool)
        source = torch.arange(batch * 4 * 256, dtype=torch.float32).reshape(
            batch, 4, 256
        ) / 1000.0
        source.requires_grad_(requires_grad)
        tokens = torch.where(valid[..., None], source, torch.zeros_like(source))
        raw_hashes = tuple((f"demo-{row}",) for row in range(batch))
        refs = tuple(
            (
                SegmentRef(raw_hashes[row][0], 0, 0, "start", False),
                SegmentRef(raw_hashes[row][0], 0, 1, "interaction", True),
                SegmentRef(raw_hashes[row][0], 1, 1, "end", False),
                None,
            )
            for row in range(batch)
        )
        return EventEncoding(tokens, valid), refs, raw_hashes, source

    @staticmethod
    def _raw_demo(content_hash):
        from icgs.data.preprocessing.events import TimedDemoInput

        return TimedDemoInput(EventSegmentationTests._transitions([0.0, 0.0]), content_hash)

    @staticmethod
    def _prepared(name, *, owner=None, embeddings=None, positions=None):
        from icgs.state.context_cache import PreparedContext

        return PreparedContext(
            [{"name": name}],
            object() if owner is None else owner,
            torch.tensor([float(len(name))]) if embeddings is None else embeddings,
            torch.tensor([float(len(name) + 1)]) if positions is None else positions,
            source_id=name,
        )

    @staticmethod
    def _lineage_inputs(ref_hashes, raw_hashes):
        from icgs.contracts.method import SegmentRef
        from icgs.models.encoders.event import EventEncoding

        event_count = len(ref_hashes)
        encoding = EventEncoding(
            torch.ones(1, event_count, 256),
            torch.ones(1, event_count, dtype=torch.bool),
        )
        refs = (tuple(
            SegmentRef(content_hash, index, index + 1, "interaction", True)
            for index, content_hash in enumerate(ref_hashes)
        ),)
        return encoding, refs, (tuple(raw_hashes),)

    def test_context_fingerprint_is_canonical_domain_separated_and_order_sensitive(self):
        from icgs.state.method_context import context_fingerprint

        raw_hashes = ("demo-a", "demo-b")
        expected_payload = {
            "domain": "icgs.event-memory",
            "schema": 1,
            "raw_hashes": list(raw_hashes),
            "encoder_id": "encoder-a",
            "segmentation_id": "segments-a",
        }
        expected = hashlib.sha256(
            json.dumps(
                expected_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        actual = context_fingerprint(raw_hashes, "encoder-a", "segments-a")
        self.assertEqual(actual, expected)
        self.assertEqual(actual, context_fingerprint(raw_hashes, "encoder-a", "segments-a"))
        self.assertNotEqual(actual, context_fingerprint(tuple(reversed(raw_hashes)), "encoder-a", "segments-a"))
        self.assertNotEqual(actual, context_fingerprint(raw_hashes, "encoder-b", "segments-a"))
        self.assertNotEqual(actual, context_fingerprint(raw_hashes, "encoder-a", "segments-b"))

        with self.assertRaisesRegex(TypeError, "raw_hashes.*tuple"):
            context_fingerprint(list(raw_hashes), "encoder-a", "segments-a")
        with self.assertRaisesRegex(ValueError, "raw_hashes.*nonempty"):
            context_fingerprint((), "encoder-a", "segments-a")
        for args, message in (
            ((("",), "encoder-a", "segments-a"), "raw_hashes"),
            ((raw_hashes, " ", "segments-a"), "encoder_id"),
            ((raw_hashes, "encoder-a", " "), "segmentation_id"),
        ):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, message):
                    context_fingerprint(*args)

    def test_event_memory_is_batched_owned_and_keeps_autograd(self):
        from icgs.state.method_context import build_event_memory, context_fingerprint

        encoding, refs, raw_hashes, source = self._memory_inputs(batch=2, requires_grad=True)
        memory = build_event_memory(
            encoding,
            refs,
            raw_hashes,
            "encoder-a",
            "segments-a",
        )
        self.assertEqual(tuple(memory.tokens.shape), (2, 4, 256))
        self.assertEqual(tuple(memory.valid.shape), (2, 4))
        self.assertEqual(memory.refs, refs)
        self.assertEqual(memory.raw_hashes, raw_hashes)
        self.assertEqual(
            memory.fingerprints,
            tuple(
                context_fingerprint(row, "encoder-a", "segments-a")
                for row in raw_hashes
            ),
        )
        self.assertIsNot(memory.tokens, encoding.tokens)
        self.assertIsNot(memory.valid, encoding.valid)
        self.assertNotEqual(memory.tokens.data_ptr(), encoding.tokens.data_ptr())
        self.assertNotEqual(memory.valid.data_ptr(), encoding.valid.data_ptr())

        memory.tokens.sum().backward()
        self.assertIsNotNone(source.grad)
        self.assertTrue(torch.isfinite(source.grad).all().item())
        snapshot = memory.tokens.detach().clone()
        with torch.no_grad():
            encoding.tokens.add_(1000.0)
            encoding.valid.zero_()
        torch.testing.assert_close(memory.tokens, snapshot)
        self.assertTrue(memory.valid.any().item())

    def test_event_memory_rejects_invalid_alignment_hashes_and_padding(self):
        from icgs.contracts.method import SegmentRef
        from icgs.models.encoders.event import EventEncoding
        from icgs.state.method_context import EventMemory, build_event_memory

        encoding, refs, raw_hashes, _ = self._memory_inputs()
        missing_ref = (tuple([None, *refs[0][1:]]),)
        with self.assertRaisesRegex(ValueError, "valid.*ref|ref.*valid"):
            build_event_memory(encoding, missing_ref, raw_hashes, "encoder-a", "segments-a")

        padded_ref = (tuple([*refs[0][:3], refs[0][1]]),)
        with self.assertRaisesRegex(ValueError, "padding|invalid.*ref|ref.*invalid"):
            build_event_memory(encoding, padded_ref, raw_hashes, "encoder-a", "segments-a")

        foreign = list(refs[0])
        foreign[1] = SegmentRef("foreign", 0, 1, "interaction", True)
        with self.assertRaisesRegex(ValueError, "hash|raw"):
            build_event_memory(
                encoding, (tuple(foreign),), raw_hashes, "encoder-a", "segments-a"
            )

        nonzero = encoding.tokens.clone()
        nonzero[0, 3, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "invalid.*zero|padding.*zero"):
            build_event_memory(
                EventEncoding(nonzero, encoding.valid),
                refs,
                raw_hashes,
                "encoder-a",
                "segments-a",
            )
        nonfinite = encoding.tokens.clone()
        nonfinite[0, 3, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            build_event_memory(
                EventEncoding(nonfinite, encoding.valid),
                refs,
                raw_hashes,
                "encoder-a",
                "segments-a",
            )
        valid_nonfinite = encoding.tokens.clone()
        valid_nonfinite[0, 1, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            build_event_memory(
                EventEncoding(valid_nonfinite, encoding.valid),
                refs,
                raw_hashes,
                "encoder-a",
                "segments-a",
            )
        with self.assertRaisesRegex(ValueError, "batch|raw_hashes"):
            build_event_memory(
                encoding, refs, (raw_hashes[0], raw_hashes[0]), "encoder-a", "segments-a"
            )
        with self.assertRaises(TypeError):
            EventMemory(
                encoding.tokens,
                encoding.valid,
                refs,
                raw_hashes,
                "encoder-a",
                "segments-a",
                fingerprints=("caller-owned",),
            )

    def test_event_memory_rejects_duplicate_raw_hashes(self):
        from icgs.state.method_context import build_event_memory

        encoding, refs, raw_hashes = self._lineage_inputs(
            ("demo-a", "demo-a"),
            ("demo-a", "demo-a"),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            build_event_memory(
                encoding, refs, raw_hashes, "encoder-a", "segments-a"
            )

    def test_event_memory_rejects_declared_but_unrepresented_demo(self):
        from icgs.state.method_context import build_event_memory

        encoding, refs, raw_hashes = self._lineage_inputs(
            ("demo-a", "demo-a"),
            ("demo-a", "demo-b"),
        )
        with self.assertRaisesRegex(ValueError, "declared.*represented|represented.*declared"):
            build_event_memory(
                encoding, refs, raw_hashes, "encoder-a", "segments-a"
            )

    def test_event_memory_rejects_demo_block_order_mismatch(self):
        from icgs.state.method_context import build_event_memory

        encoding, refs, raw_hashes = self._lineage_inputs(
            ("demo-b", "demo-b", "demo-a", "demo-a"),
            ("demo-a", "demo-b"),
        )
        with self.assertRaisesRegex(ValueError, "block.*order|order.*block"):
            build_event_memory(
                encoding, refs, raw_hashes, "encoder-a", "segments-a"
            )

    def test_event_memory_rejects_split_demo_blocks(self):
        from icgs.state.method_context import build_event_memory

        encoding, refs, raw_hashes = self._lineage_inputs(
            ("demo-a", "demo-b", "demo-a", "demo-b"),
            ("demo-a", "demo-b"),
        )
        with self.assertRaisesRegex(ValueError, "block.*order|order.*block"):
            build_event_memory(
                encoding, refs, raw_hashes, "encoder-a", "segments-a"
            )

    def test_method_context_derives_window_validity_and_preserves_injected_objects(self):
        from icgs.state.method_context import MethodContext, build_event_memory

        encoding, refs, raw_hashes, _ = self._memory_inputs()
        events = build_event_memory(encoding, refs, raw_hashes, "encoder-a", "segments-a")
        raw_demos = (self._raw_demo("demo-0"),)
        full = self._prepared("full", owner=object())
        window = self._prepared("window", owner=object())
        windows = (None, window, None, None)
        full_buffers = (full.embeddings, full.positions)
        window_buffers = (window.embeddings, window.positions)

        context = MethodContext(raw_demos, events, full, windows, "reference-a")
        self.assertIsNot(full.owner, window.owner)
        self.assertIs(context.raw_demos, raw_demos)
        self.assertIs(context.events, events)
        self.assertIs(context.native_full, full)
        self.assertIs(context.native_windows, windows)
        self.assertIs(context.native_windows[1], window)
        self.assertIs(full.embeddings, full_buffers[0])
        self.assertIs(full.positions, full_buffers[1])
        self.assertIs(window.embeddings, window_buffers[0])
        self.assertIs(window.positions, window_buffers[1])
        torch.testing.assert_close(
            context.native_window_valid,
            torch.tensor([[False, True, False, False]], dtype=torch.bool),
        )

        absent = MethodContext(raw_demos, events, full, (None,) * 4, "reference-a")
        self.assertFalse(absent.native_window_valid.any().item())
        self.assertNotIn("native_window_valid", MethodContext.__dataclass_fields__)
        other_reference = MethodContext(raw_demos, events, full, windows, "reference-b")
        self.assertEqual(other_reference.events.fingerprints, context.events.fingerprints)

    def test_method_context_rejects_window_placement_length_and_missing_full(self):
        from icgs.state.method_context import MethodContext, build_event_memory

        encoding, refs, raw_hashes, _ = self._memory_inputs()
        events = build_event_memory(encoding, refs, raw_hashes, "encoder-a", "segments-a")
        demos = (self._raw_demo("demo-0"),)
        full = self._prepared("full")
        window = self._prepared("window")
        with self.assertRaisesRegex(ValueError, "native_windows.*L|length"):
            MethodContext(demos, events, full, (None,) * 3, "reference-a")
        with self.assertRaisesRegex(ValueError, "structural|interaction|event"):
            MethodContext(demos, events, full, (window, None, None, None), "reference-a")
        with self.assertRaisesRegex(ValueError, "structural|interaction|event|padding"):
            MethodContext(demos, events, full, (None, None, None, window), "reference-a")
        with self.assertRaisesRegex(TypeError, "native_full|PreparedContext"):
            MethodContext(demos, events, None, (None,) * 4, "reference-a")

    def test_method_context_rejects_context_and_mutable_buffer_aliases(self):
        from icgs.state.method_context import MethodContext, build_event_memory

        encoding, refs, raw_hashes, _ = self._memory_inputs()
        events = build_event_memory(encoding, refs, raw_hashes, "encoder-a", "segments-a")
        demos = (self._raw_demo("demo-0"),)
        full = self._prepared("full")
        with self.assertRaisesRegex(ValueError, "alias|distinct"):
            MethodContext(demos, events, full, (None, full, None, None), "reference-a")

        shared = torch.ones(2)
        full_shared = self._prepared("full-shared", embeddings=shared)
        window_shared = self._prepared("window-shared", embeddings=shared)
        with self.assertRaisesRegex(ValueError, "alias|buffer"):
            MethodContext(
                demos,
                events,
                full_shared,
                (None, window_shared, None, None),
                "reference-a",
            )

        positions = torch.arange(4.0)
        full_view = self._prepared("full-view", positions=positions[:2])
        window_view = self._prepared("window-view", positions=positions[1:3])
        with self.assertRaisesRegex(ValueError, "alias|buffer"):
            MethodContext(
                demos,
                events,
                full_view,
                (None, window_view, None, None),
                "reference-a",
            )

    def test_method_context_requires_online_batch_and_exact_raw_demo_order(self):
        import icgs.state.method_context as context_module
        from icgs.contracts.method import SegmentRef
        from icgs.models.encoders.event import EventEncoding
        from icgs.state.method_context import MethodContext, build_event_memory

        batched_encoding, batched_refs, batched_hashes, _ = self._memory_inputs(batch=2)
        batched = build_event_memory(
            batched_encoding,
            batched_refs,
            batched_hashes,
            "encoder-a",
            "segments-a",
        )
        with self.assertRaisesRegex(ValueError, "B=1|batch"):
            MethodContext(
                (self._raw_demo("demo-0"),),
                batched,
                self._prepared("full-batched"),
                (None,) * 4,
                "reference-a",
            )

        valid = torch.ones(1, 6, dtype=torch.bool)
        tokens = torch.ones(1, 6, 256)
        hashes = (("demo-a", "demo-b"),)
        refs = ((
            SegmentRef("demo-a", 0, 0, "start", False),
            SegmentRef("demo-a", 0, 1, "interaction", True),
            SegmentRef("demo-a", 1, 1, "end", False),
            SegmentRef("demo-b", 0, 0, "start", False),
            SegmentRef("demo-b", 0, 1, "interaction", True),
            SegmentRef("demo-b", 1, 1, "end", False),
        ),)
        events = build_event_memory(
            EventEncoding(tokens, valid), refs, hashes, "encoder-a", "segments-a"
        )
        full = self._prepared("full-ordered")
        demos = (self._raw_demo("demo-a"), self._raw_demo("demo-b"))
        MethodContext(demos, events, full, (None,) * 6, "reference-a")
        with self.assertRaisesRegex(ValueError, "order|raw.*hash"):
            MethodContext(tuple(reversed(demos)), events, full, (None,) * 6, "reference-a")

        source = inspect.getsource(context_module)
        for forbidden in (
            "TaskState",
            "InstantPolicy",
            "router",
            "branch_copy(",
            "deepcopy(",
            ".prepare(",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
