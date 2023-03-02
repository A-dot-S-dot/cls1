from unittest import TestCase

import numpy as np
from finite_volume.shallow_water.solver.mcl import *
import finite_volume.shallow_water as swe
from numpy.testing import assert_equal


class TestMcl(TestCase):
    def test_limiter(self):
        value_left = np.array([[2.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
        value_right = np.array([[2.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        limiter = MCLFlux(gravitational_acceleration=1.0)

        flux_left, flux_right = limiter(value_left, value_right)
        expected_flux = np.array([[0.0, 2.0], [0.0, 1.0], [0.0, 0.5]])

        assert_equal(flux_left, -expected_flux)
        assert_equal(flux_right, expected_flux)
