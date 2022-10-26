from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import numpy as np
import pde_solver.system_vector as vector
from pde_solver.discretization import DiscreteSolution
from pde_solver.discretization.finite_volume import FiniteVolumeSpace
from pde_solver.mesh import Interval, UniformMesh


class TestNaturalSourceTermDiscretization(TestCase):
    step_length = 0.5
    left_height = [1]
    right_height = [4]
    topography_step = [1]
    expected_source_term = [5]

    def test_numerical_flux(self):
        for i in range(len(self.expected_source_term)):
            self.assertAlmostEqual(
                vector.calculate_natural_source_term_discretization(
                    self.left_height[i],
                    self.right_height[i],
                    self.topography_step[i],
                    self.step_length,
                ),
                self.expected_source_term[i],
                msg=f"right flux, index={i}",
            )


class TestWetDryPreservingSourceTermDiscretization(TestNaturalSourceTermDiscretization):
    step_length = 0.5
    left_height = [1, 1]
    right_height = [4, 4]
    topography_step = [1, -1]
    expected_source_term = [5, -5]

    def test_numerical_flux(self):
        for i in range(len(self.expected_source_term)):
            self.assertAlmostEqual(
                vector.calculate_wet_dry_preserving_source_term_discretization(
                    self.left_height[i],
                    self.right_height[i],
                    self.topography_step[i],
                    self.step_length,
                ),
                self.expected_source_term[i],
                msg=f"right flux, index={i}",
            )


class TestIntermediateVelocities(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    volume_space = FiniteVolumeSpace(mesh)
    velocities = vector.ShallowWaterIntermediateVelocities(volume_space, 1)
    expected_left_velocities = np.array([-2, -2])
    expected_right_velocities = np.array([3, 3])

    def test_numerical_flux(self):
        self.velocities.update(np.array([[1, -1], [4, 4]]))

        for i in range(self.volume_space.node_number):
            self.assertAlmostEqual(
                self.velocities.left_velocities[i],
                self.expected_left_velocities[i],
                msg=f"left flux, index={i}",
            )
            self.assertAlmostEqual(
                self.velocities.right_velocities[i],
                self.expected_right_velocities[i],
                msg=f"right flux, index={i}",
            )


class TestGodunovNodeFluxesCalculator(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    volume_space = FiniteVolumeSpace(mesh)
    discrete_solution = DiscreteSolution(
        0, np.array([[1, -1], [4, 4]]), np.array([0, 0.5])
    )

    velocities = vector.ShallowWaterIntermediateVelocities(volume_space, 1)
    calculator = vector.ShallowWaterGodunovNodeFluxesCalculator(
        volume_space,
        1,
        np.array([0, 1]),
        velocities,
        vector.calculate_natural_source_term_discretization,
    )
    calculator(discrete_solution.end_values, 1)

    def test_cell_indices(self):
        self.assertTupleEqual(self.calculator.cell_indices, (0, 1))

    def test_swe_values(self):
        for i in range(2):
            self.assertListEqual(
                list(self.calculator._swe_values[i]),
                list(self.discrete_solution.end_values[i]),
            )

    def test_heights(self):
        self.assertTupleEqual(self.calculator._heights, (1, 4))

    def test_dischares(self):
        self.assertTupleEqual(self.calculator._discharges, (-1, 4))

    def test_fluxes(self):
        expected_fluxes = [[-1, 1.5], [4, 12]]
        for i in range(2):
            self.assertListEqual(list(self.calculator._fluxes[i]), expected_fluxes[i])

    def test_wave_velocities(self):
        self.assertTupleEqual(self.calculator._wave_velocities, (-2, 3))

    def test_cell_fluxes(self):
        expected_cell_fluxes = [[-3.8, 0.7], [-3.8, -1.8]]

        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    self.calculator.node_fluxes[i][j], expected_cell_fluxes[i][j]
                )

    def test_topography_step(self):
        self.assertEqual(self.calculator._topography_step, 1)


class TestSWEGodunovFlux(TestCase):
    discrete_solution = DiscreteSolution(
        0, np.array([[1, 1], [0, 0], [1, -1], [2, 0]]), np.array([0, 0.25, 0.5, 0.75])
    )

    velocities = vector.ShallowWaterIntermediateVelocities(VOLUME_SPACE, 1)
    calculator = vector.ShallowWaterGodunovNodeFluxesCalculator(
        VOLUME_SPACE,
        1,
        np.array([1, 1, 1, 1]),
        velocities,
        vector.calculate_natural_source_term_discretization,
    )

    numerical_flux = vector.ShallowWaterGodunovNumericalFlux(VOLUME_SPACE, calculator)

    expected_left_flux = np.array(
        [[0.0, 2.0], [0.0, 2.0], [-1.41421356, 1.70710678], [-1.41421356, 0.87867966]]
    )
    expected_right_flux = np.array(
        [[0.0, 2.0], [-1.41421356, 1.70710678], [-1.41421356, 0.87867966], [0.0, 2.0]]
    )

    def test_numerical_flux(self):
        left_flux, right_flux = self.numerical_flux(self.discrete_solution.end_values)

        for i in range(VOLUME_SPACE.dimension):
            for j in range(2):
                self.assertAlmostEqual(
                    left_flux[i, j],
                    self.expected_left_flux[i, j],
                    msg=f"left flux, index={i}",
                )
                self.assertAlmostEqual(
                    right_flux[i, j],
                    self.expected_right_flux[i, j],
                    msg=f"right flux, index={i}",
                )
