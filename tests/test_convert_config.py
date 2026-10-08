import unittest

from tkintegers_convert import ConversionConfig


class ConversionConfigTests(unittest.TestCase):
    def test_config_serializes_selected_scale_policy(self):
        config = ConversionConfig(weight_k=8, weights_per_scale=32, scale_type="int8-linear")
        data = config.to_dict()
        self.assertEqual(data["weight_k"], 8)
        self.assertEqual(data["weights_per_scale"], 32)
        self.assertEqual(data["scale"]["dtype"], "int8")

    def test_config_accepts_independent_ternary_scale_k(self):
        config = ConversionConfig(weight_k=3, scale_type="ternary-log", scale_k=13)
        self.assertEqual(config.to_dict()["scale"]["k"], 13)


if __name__ == "__main__":
    unittest.main()
