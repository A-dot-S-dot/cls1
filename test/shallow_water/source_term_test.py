from unittest import TestCase

import numpy as np

import shallow_water


class TestNaturalSourceTerm(TestCase):
    def test_numerical_flux(self):
        source_term = shallow_water.NaturalSouceTerm(0.5)

        left_height = np.array([1, 4])
        right_height = np.array([4, 1])
        topography_step = np.array([1, -1])

        expected_discretization = np.array([5, -5])
        discretization = source_term(left_height, right_height, topography_step)

        self.assertTrue(np.array_equal(discretization, expected_discretization))
