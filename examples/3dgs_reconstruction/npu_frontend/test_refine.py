import unittest
import numpy as np
from backends import NumpyBackend
from matcher import match, scalar_reference
from refine import RefinedBackend


class Refine(unittest.TestCase):
    def test_bounded_adversarial_scores_preserve_matches(self):
        rng = np.random.default_rng(832)
        a = rng.integers(0, 60, (65, 128), dtype=np.uint8)
        b = np.concatenate((a, a[:3], rng.integers(0, 60, (17, 128), dtype=np.uint8)))
        class Perturbed(NumpyBackend):
            def dot(self, left, right):
                result = super().dot(left, right)
                noise = rng.uniform(-.003, .003, result.shape).astype(np.float32)
                return np.maximum(result+noise, 0)
        for tile in (32, 64, 128):
            backend = RefinedBackend(Perturbed())
            got, _ = match(a, b, backend, tile)
            np.testing.assert_array_equal(got, scalar_reference(a, b))
        with self.assertRaises(ValueError):
            RefinedBackend(NumpyBackend(), float("nan"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
