import unittest

try:
    import torch
    from base3_pack import pack_ktrit, unpack_ktrit
    from tk_s12 import encode, decode, pack12, unpack12
    from tk_codec import pack_hi, unpack_hi
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "codec tests require torch")
class CodecTests(unittest.TestCase):
    def test_any_k_base3_round_trip(self):
        generator = torch.Generator().manual_seed(7)
        for k in (1, 2, 3, 5, 8, 10, 16):
            levels = (3 ** k - 1) // 2
            codes = torch.randint(-levels, levels + 1, (3, 20), generator=generator, dtype=torch.int32)
            packed, width, _ = pack_ktrit(codes, k)
            self.assertTrue(torch.equal(unpack_ktrit(packed, width, k), codes))

    def test_s12_round_trip_and_zero(self):
        values = torch.tensor([0.0, 1e-3, 0.1, 1.0], dtype=torch.float64)
        codes = encode(values)
        self.assertTrue(torch.equal(unpack12(pack12(codes), codes.numel()), codes))
        self.assertEqual(int(codes[0]), 0)
        self.assertEqual(float(decode(codes)[0]), 0.0)

    def test_integer_anchor_round_trip(self):
        values = torch.tensor([-0.5, 0.0, 0.25, 1.0])
        codes, meta = pack_hi(values, 15)
        self.assertFalse(codes.dtype.is_floating_point)
        self.assertFalse(meta.dtype.is_floating_point)
        self.assertEqual(unpack_hi(codes, meta).shape, values.shape)


if __name__ == "__main__":
    unittest.main()
