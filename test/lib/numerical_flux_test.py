from typing import Tuple
from unittest import TestCase

import numpy as np

from lib.numerical_flux import *


class TestNumericalFlux:
    def __call__(
        self, time: float, dof_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return dof_vector, np.roll(dof_vector, -1, axis=0)


class TestNumericalFluxWithHistory(TestCase):
    def test_left_numerical_flux_with_history(self):
        vector = np.array([[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]])
        flux = NumericalFluxWithHistory(TestNumericalFlux())

        flux(0, vector)
        flux(1, vector)

        expected_flux_left_history = [
            [[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]],
            [[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]],
        ]

        for i in range(2):
            for j in range(4):
                self.assertListEqual(
                    list(flux.flux_left_history[i, j]), expected_flux_left_history[i][j]
                )

    def test_right_numerical_flux_with_history(self):
        vector = np.array([[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]])
        flux = NumericalFluxWithHistory(TestNumericalFlux())

        flux(0, vector)
        flux(1, vector)

        expected_flux_right_history = [
            [[4.0, 2.0], [6.0, 4.0], [8.0, 6.0], [0.0, 0.0]],
            [[4.0, 2.0], [6.0, 4.0], [8.0, 6.0], [0.0, 0.0]],
        ]

        for i in range(2):
            for j in range(4):
                self.assertListEqual(
                    list(flux.flux_right_history[i, j]),
                    expected_flux_right_history[i][j],
                )


class TestSubgridFlux(TestCase):
    def test_subgrid_flux_left(self):
        subgrid_flux = SubgridFlux(TestNumericalFlux(), TestNumericalFlux(), 2)
        vector = np.array([[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]])
        expected_subgrid_flux_left = [[-2.0, -1.0], [-1.0, -1.0]]

        subgrid_flux_left, _ = subgrid_flux(0, vector)

        for i in range(2):
            self.assertListEqual(
                list(subgrid_flux_left[i]), expected_subgrid_flux_left[i]
            )

    def test_subgrid_flux_right(self):
        subgrid_flux = SubgridFlux(TestNumericalFlux(), TestNumericalFlux(), 2)
        vector = np.array([[0.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]])
        expected_subgrid_flux_right = [[-1.0, -1.0], [-2.0, -1.0]]

        _, subgrid_flux_right = subgrid_flux(0, vector)

        for i in range(2):
            self.assertListEqual(
                list(subgrid_flux_right[i]), expected_subgrid_flux_right[i]
            )
