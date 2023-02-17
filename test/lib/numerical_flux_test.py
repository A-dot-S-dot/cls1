from typing import Tuple
from unittest import TestCase

import numpy as np
import shallow_water
from core import finite_volume
from numpy.testing import assert_almost_equal, assert_equal

from lib.numerical_flux import *


class TestNumericalFlux(NumericalFlux):
    input_dimension = 2

    def __call__(self, vector_left, vector_right) -> Tuple[np.ndarray, np.ndarray]:
        return vector_left, vector_right


class TestNumericalFluxWithHistory(TestCase):
    def test_numerical_flux_with_history(self):
        vector = np.array([[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]])
        flux = NumericalFluxWithHistory(TestNumericalFlux())
        neighbours_builder = finite_volume.NodeNeighboursWithPeriodicBoundary(1)

        flux(*neighbours_builder(vector))
        flux(*neighbours_builder(vector))

        expected_flux_left_history = np.array(
            [
                [[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0]],
                [[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0]],
            ]
        )
        expected_flux_right_history = np.array(
            [
                [[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]],
                [[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]],
            ]
        )

        assert_equal(expected_flux_left_history, flux.flux_left_history)
        assert_equal(expected_flux_right_history, flux.flux_right_history)


class TestNumericalFluxWithArbitraryInput(TestCase):
    flux = NumericalFluxWithArbitraryInput(TestNumericalFlux())
    vector = np.array([[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]])
    expected_flux_left = np.array([[8.0, 6.0], [0.0, 0.0], [4.0, 2.0], [6.0, 4.0]])
    expected_flux_right = np.array([[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]])

    def test_exact_input(self):
        neighbours_builder = finite_volume.NodeNeighboursWithPeriodicBoundary(1)
        fl, fr = self.flux(*neighbours_builder(self.vector))

        assert_equal(fl, self.expected_flux_left)
        assert_equal(fr, self.expected_flux_right)

    def test_too_large_input(self):
        neighbours_builder = finite_volume.NodeNeighboursWithPeriodicBoundary(2)
        fl, fr = self.flux(*neighbours_builder(self.vector))

        assert_equal(fl, self.expected_flux_left)
        assert_equal(fr, self.expected_flux_right)


class TestSubgridFlux(TestCase):
    def test_subgrid_flux_left(self):
        subgrid_flux = SubgridFlux(TestNumericalFlux(), TestNumericalFlux(), 2)
        vector = np.array([[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]])
        coarse_vector = np.array([[2.0, 1.0], [7.0, 5.0]])
        neighbours = finite_volume.NodeNeighboursWithPeriodicBoundary(1)

        expected_subgrid_flux_left = np.array([[1.0, 1.0], [2.0, 1.0]])
        expected_subgrid_flux_right = np.array([[2.0, 1.0], [1.0, 1.0]])

        subgrid_flux_left, subgrid_flux_right = subgrid_flux(
            neighbours(vector), neighbours(coarse_vector)
        )
        assert_equal(subgrid_flux_left, expected_subgrid_flux_left)
        assert_equal(subgrid_flux_right, expected_subgrid_flux_right)


class TestLaxFriedrichFlux(TestCase):
    def test_flux(self):
        value_left = np.array(
            [[2.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]]
        )
        value_right = np.array(
            [[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0], [1.0, 1.0]]
        )

        riemann_solver = shallow_water.RiemannSolver(1.0)
        numerical_flux = LaxFriedrichsFlux(riemann_solver)
        flux_left, flux_right = numerical_flux(value_left, value_right)

        expected_flux_left = np.array(
            [[1.5, 0.75], [1.5, 1.75], [-1.5, 1.75], [-1.5, 0.75]]
        )
        expected_flux_right = -np.array(
            [[1.5, 1.75], [-1.5, 1.75], [-1.5, 0.75], [1.5, 0.75]]
        )

        assert_almost_equal(flux_left, expected_flux_left)
        assert_almost_equal(flux_right, expected_flux_right)
