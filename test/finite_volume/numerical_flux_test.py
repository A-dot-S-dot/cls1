from unittest import TestCase

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

import finite_volume.shallow_water as swe
from finite_volume.numerical_flux import *


class TestNumericalFlux(NumericalFlux):
    input_dimension = 2

    def __call__(self, values_left, values_right):
        return -values_left, values_left


class TestGetDofVector(NumericalFlux):
    def test_two_inputs(self):
        value_left = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        value_right = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

        expected_value = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        assert_equal(get_dof_vector(value_left, value_right), expected_value)

    def test_four_inputs(self):
        value_Left = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        value_left = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        value_right = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        value_Right = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])

        expected_value = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]
        )
        assert_equal(
            get_dof_vector(value_Left, value_left, value_right, value_Right),
            expected_value,
        )


class TestNumericalFluxWithHistory(TestCase):
    def test_numerical_flux_with_history(self):
        values_left = np.array(
            [[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]]
        )
        values_right = np.array(
            [[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0], [0.0, 0.0]]
        )
        flux = NumericalFluxWithHistory(TestNumericalFlux())

        flux(values_left, values_right)
        flux(values_left, values_right)

        expected_flux_left_history = -np.array(
            [
                [[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]],
                [[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]],
            ]
        )
        expected_flux_right_history = np.array(
            [
                [[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]],
                [[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]],
            ]
        )

        assert_equal(expected_flux_left_history, flux.flux_left_history)
        assert_equal(expected_flux_right_history, flux.flux_right_history)


class TestNumericalFluxWithArbitraryInput(TestCase):
    flux = NumericalFluxWithArbitraryInput(TestNumericalFlux())
    values_left = np.array([[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]])
    values_right = np.array(
        [[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0], [0.0, 0.0]]
    )
    expected_flux_left = -np.array(
        [[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]]
    )
    expected_flux_right = np.array(
        [[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]]
    )

    def test_exact_input(self):
        fl, fr = self.flux(self.values_left, self.values_right)

        assert_equal(fl, self.expected_flux_left)
        assert_equal(fr, self.expected_flux_right)

    def test_too_large_input(self):
        fl, fr = self.flux(
            self.values_left, self.values_left, self.values_right, self.values_right
        )

        assert_equal(fl, self.expected_flux_left)
        assert_equal(fr, self.expected_flux_right)


# class TestSubgridFlux(TestCase):
#     def test_subgrid_flux_left(self):
#         subgrid_flux = SubgridFlux(TestNumericalFlux(), TestNumericalFlux(), 2)
#         values_left = np.array(
#             [[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]]
#         )
#         values_right = np.array(
#             [[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0], [0.0, 0.0]]
#         )
#         coarse_values_left = np.array([[7.0, 5.0], [2.0, 1.0], [7.0, 5.0]])
#         coarse_values_right = np.array([[2.0, 1.0], [7.0, 5.0], [2.0, 1.0]])

#         expected_subgrid_flux_left = np.array([[1.0, 1.0], [2.0, 1.0]])
#         expected_subgrid_flux_right = np.array([[2.0, 1.0], [1.0, 1.0]])

#         subgrid_flux_left, subgrid_flux_right = subgrid_flux(
#             (values_left, values_right), (coarse_values_left, coarse_values_right)
#         )
#         assert_equal(subgrid_flux_left, expected_subgrid_flux_left)
#         assert_equal(subgrid_flux_right, expected_subgrid_flux_right)


class TestLaxFriedrichFlux(TestCase):
    def test_flux(self):
        value_left = np.array(
            [[2.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]]
        )
        value_right = np.array(
            [[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0], [1.0, 1.0]]
        )

        riemann_solver = swe.RiemannSolver(1.0)
        numerical_flux = LaxFriedrichsFlux(riemann_solver)
        flux_left, flux_right = numerical_flux(value_left, value_right)

        expected_flux = np.array(
            [[1.5, 0.75], [1.5, 1.75], [-1.5, 1.75], [-1.5, 0.75], [1.5, 0.75]]
        )

        assert_almost_equal(flux_left, -expected_flux)
        assert_almost_equal(flux_right, expected_flux)
