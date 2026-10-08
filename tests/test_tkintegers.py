import unittest

from tkintegers import (
    ContainerSpec,
    density_ladder,
    mixture_stats,
    preset_specs,
    resolve_scale_policy,
    scale_policy_choices,
)


class TkIntegerMathTests(unittest.TestCase):
    def test_s12_ideal_density_values(self):
        self.assertAlmostEqual(ContainerSpec(1, 20, 12).ideal_bpw, 2.2)
        self.assertAlmostEqual(ContainerSpec(1, 15, 12).ideal_bpw, 2.4)
        self.assertAlmostEqual(ContainerSpec(1, 12, 12).ideal_bpw, 2.6)

    def test_finite_payload_accounts_for_byte_padding(self):
        g20 = ContainerSpec(1, 20, 12).storage(2_000)
        g15 = ContainerSpec(1, 15, 12).storage(2_000)
        self.assertEqual(g20["payload_bytes"], 550)
        self.assertEqual(g15["payload_bytes"], 601)
        self.assertGreater(g15["payload_bpw"], g20["payload_bpw"])

    def test_residual_is_added_as_a_second_plane(self):
        spec = ContainerSpec(5, 128, 10, residual_k=3, residual_weights_per_scale=64, residual_scale_bits=10)
        stats = spec.storage(1_024)
        self.assertEqual(stats["base"]["groups"], 8)
        self.assertEqual(stats["residual"]["groups"], 16)
        self.assertGreater(stats["ideal_bpw"], ContainerSpec(5, 128, 10).ideal_bpw)

    def test_density_ladder_includes_every_intermediate_group(self):
        rows = density_ladder(1, 12)
        self.assertEqual(len(rows), 128)
        self.assertEqual(rows[14]["group"], 15)
        self.assertEqual(rows[19]["group"], 20)

    def test_mixture_is_weighted_by_actual_parameters(self):
        parts = [
            {"label": "small", "weights": 100, "spec": ContainerSpec(1, 20, 12)},
            {"label": "large", "weights": 900, "spec": ContainerSpec(5, 128, 10)},
        ]
        stats = mixture_stats(parts)
        self.assertEqual(stats["weights"], 1_000)
        self.assertEqual(len(stats["parts"]), 2)
        self.assertGreater(stats["payload_bpw"], 0.0)

    def test_presets_include_canonical_k_choices_and_s12(self):
        names = {spec.name for spec in preset_specs()}
        self.assertIn("TK S12 / g20", names)
        self.assertIn("TK k5 / g128 / s10", names)
        self.assertIn("TK k15 / g128 / s10", names)

    def test_scale_policy_choices_cover_integer_and_float_families(self):
        keys = {choice.key for choice in scale_policy_choices()}
        self.assertIn("existing-k10-log", keys)
        self.assertIn("tk-s12", keys)
        self.assertIn("int8-linear", keys)
        self.assertIn("bf16", keys)
        self.assertAlmostEqual(resolve_scale_policy("ternary-log", 8).bits, 12.8)

    def test_container_spec_records_a_non_custom_scale_policy(self):
        spec = ContainerSpec(5, 128, 0, scale_type="ternary-log", scale_k=8)
        self.assertEqual(spec.scale_type, "ternary-log")
        self.assertEqual(spec.scale_k, 8)
        self.assertAlmostEqual(spec.scale_bits, 12.8)


if __name__ == "__main__":
    unittest.main()
