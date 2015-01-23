import unittest
import numpy as np
from hindemith.types.hmarray import hmarray


class LayerTest(unittest.TestCase):
    def _check(self):
        try:
            np.testing.assert_allclose(self.actual, self.expected, rtol=1e-04)
        except AssertionError as e:
            self.fail(e)

    def setUp(self):
        self.in_batch = np.random.rand(5, 3, 256, 256).astype(np.float32) * 255
