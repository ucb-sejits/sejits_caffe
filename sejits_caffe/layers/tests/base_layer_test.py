import unittest
import numpy as np
from hindemith.types.hmarray import hmarray


class LayerTest(unittest.TestCase):
    def _check(self):
        try:
            self.actual.copy_to_host_if_dirty()
            np.testing.assert_allclose(self.actual, self.expected)
        except AssertionError as e:
            self.fail(e)

    def setUp(self):
        self.in_batch = hmarray(np.random.rand(5, 3, 256, 256).astype(np.float32))
        self.actual = hmarray(np.random.rand(5, 3, 256, 256).astype(np.float32))
        self.expected = hmarray(np.random.rand(5, 3, 256, 256).astype(np.float32))
