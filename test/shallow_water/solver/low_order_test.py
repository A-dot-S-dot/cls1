from unittest import TestCase

import numpy as np
import shallow_water
from numpy.testing import assert_almost_equal, assert_array_compare
from shallow_water.solver.low_order import *


class TestHeightBarStateModifier(TestCase):
    def test_positivity(self):
        modifier = HeightModifier()
        hl, hr = modifier.modify(np.array([1, 2, 3, 0]), np.array([-10, 10, -10, 10]))

        self.assertTrue((hl >= 0).all())
        self.assertTrue((hr >= 0).all())
