from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from shallow_water.core import *


class TestFlux(TestCase):
    def test_flux(self):
        flux = Flux(1)
        dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
        expected_flux = np.array([[1.0, 1.5], [0.0, 0.0], [-1.0, 1.5], [0.0, 2.0]])

        assert_equal(flux(dof_vector), expected_flux)

    def test_negative_height_error(self):
        self.assertRaises(
            NegativeHeightError,
            Flux(1),
            np.array([[-1, 2], [2, 3]]),
        )


class TestAverage(TestCase):
    def test_scalar_average(self):
        value_left = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
        value_right = np.array([1.0, 0.0, 1.0, 2.0, 1.0])
        expected_output = [1.5, 0.5, 0.5, 1.5, 1.5]

        assert_equal(get_average(value_left, value_right), expected_output)

    def test_system_average(self):
        value_left = np.array(
            [[2.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]]
        )
        value_right = np.array(
            [[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0], [1.0, 1.0]]
        )
        expected_output = [[1.5, 0.5], [0.5, 0.5], [0.5, -0.5], [1.5, -0.5], [1.5, 0.5]]

        assert_equal(get_average(value_left, value_right), expected_output)


class TestNullify(TestCase):
    def test_nullify(self):
        dof_vector = np.array([[0.5, 2], [0.5, 0.2], [2, 0.5], [2, 2]])
        expected_output = np.array([[0, 0], [0, 0], [2, 0.5], [2, 2]])

        assert_equal(nullify(dof_vector, eps=1.0), expected_output)


class TestGetVelocity(TestCase):
    def test_transformer(self):
        dof_vector = np.array([[2.0, -4.0], [2.0, 4.0], [0.0, 0.0]])
        expected_output = np.array([-2.0, 2.0, 0.0])

        assert_almost_equal(get_velocity(dof_vector), expected_output)


class TestIsConstant(TestCase):
    def test_not_constant(self):
        bottom = np.array([0.0, 0.0, 1.0, 0.0])
        self.assertFalse(is_constant(bottom))

    def test_is_constant(self):
        bottom = np.array([0.0, 0.0, 0.0, 0.0])
        self.assertTrue(is_constant(bottom))
