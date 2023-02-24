from unittest import TestCase

import numpy as np
from finite_volume.shallow_water.solver.low_order import *
from numpy.testing import assert_equal


class TestLowOrder(TestCase):
    values_left = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    values_right = np.array([[1.0, 1.0], [2.0, 0.0], [2.0, 0.0]])
    flux = LowOrderFlux(1, bathymetry=np.array([1.0, 0.0]))
    flux_left, flux_right = flux(values_left, values_right)

    def test_height_modification(self):
        expected_hl = np.array([1.0, 1.25, 2.0])
        expected_hr = np.array([1, 2.25, 2.0])

        assert_equal(self.flux._modified_height_left, expected_hl)
        assert_equal(self.flux._modified_height_right, expected_hr)

    def test_discharge_modification(self):
        expected_ql = np.array([1.0, 1 / 2, 0.0])
        expected_qr = np.array([1.0, 1.0, 0.0])

        assert_equal(self.flux._modified_discharge_left, expected_ql)
        assert_equal(self.flux._modified_discharge_right, expected_qr)

    def test_fluxes(self):
        expected_flux_left = np.array([[-1.0, -1.5], [-0.5, -2.5], [0.0, -2.0]])
        expected_flux_right = np.array([[1.0, 1.5], [0.5, 4.0], [0.0, 2.0]])

        assert_equal(self.flux_left, expected_flux_left)
        assert_equal(self.flux_right, expected_flux_right)
