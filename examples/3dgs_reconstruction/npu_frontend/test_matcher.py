import unittest
import numpy as np
from matcher import match, scalar_reference
from backends import NumpyBackend


class Matching(unittest.TestCase):
    def test_empty_single_zero_and_tail(self):
        rng = np.random.default_rng(931)
        cases = [(0, 0), (0, 5), (8, 0), (1, 1), (3, 1), (33, 67), (129, 131)]
        for n, m in cases:
            with self.subTest(n=n, m=m):
                a = rng.integers(0, 70, (n, 128), dtype=np.uint8)
                b = rng.integers(0, 70, (m, 128), dtype=np.uint8)
                shared = min(n, m)
                b[:shared] = a[:shared]
                expected = scalar_reference(a, b)
                for tile in (32, 64, 128):
                    got, stats = match(a, b, NumpyBackend(), tile)
                    np.testing.assert_array_equal(got, expected)
                    self.assertFalse(stats["npu_executed"])
        zeros = np.zeros((33, 128), np.uint8)
        self.assertEqual(len(match(zeros, zeros, NumpyBackend(), 32)[0]), 0)

    def test_duplicate_tie_is_rejected_across_blocks(self):
        a = np.full((1, 128), 45, np.uint8)
        b = np.zeros((65, 128), np.uint8)
        b[0] = a[0]
        b[64] = a[0]
        self.assertEqual(len(match(a, b, NumpyBackend(), 32, cross_check=False)[0]), 0)

    def test_reverse_ratio_is_required_for_cross_check(self):
        b = np.full((1, 128), 45, np.uint8)
        a = np.repeat(b, 2, axis=0)
        self.assertEqual(len(match(a, b, NumpyBackend(), cross_check=False)[0]), 2)
        self.assertEqual(len(match(a, b, NumpyBackend(), cross_check=True)[0]), 0)

    def test_invalid_input_and_corrupt_backend_fail(self):
        with self.assertRaises(ValueError):
            match(np.zeros((3, 128), np.float32), np.zeros((3, 128), np.uint8), NumpyBackend())
        class Broken(NumpyBackend):
            def dot(self, a, b):
                return np.full((1, 1, 128, 128), np.nan, np.float32)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            match(np.zeros((3, 128), np.uint8), np.zeros((3, 128), np.uint8), Broken())


if __name__ == "__main__":
    unittest.main(verbosity=2)
