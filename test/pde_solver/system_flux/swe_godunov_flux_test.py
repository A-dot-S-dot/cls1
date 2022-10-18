from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import numpy as np
from pde_solver.discrete_solution.observed_discrete_solution import (
    DiscreteSolutionObservable,
)
from pde_solver.mesh import Interval, UniformMesh
from pde_solver.solver_space import FiniteVolumeSpace
from pde_solver.system_flux import (
    GodunovCellFluxesCalculator,
    SWEGodunovNumericalFlux,
    SWEIntermediateVelocities,
    NaturalSourceTermDiscretization,
)
from pde_solver.system_flux.swe_intermediate_velocities import SWEIntermediateVelocities


class TestGodunovCellFluxesCalculator(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    volume_space = FiniteVolumeSpace(mesh)
    discrete_solution = DiscreteSolutionObservable(0, np.array([[1, -1], [4, 4]]))

    velocities = SWEIntermediateVelocities(discrete_solution)
    velocities.volume_space = volume_space
    velocities.gravitational_acceleration = 1
    velocities.update()

    source_term = NaturalSourceTermDiscretization()
    source_term.step_length = 0.5

    calculator = GodunovCellFluxesCalculator()
    calculator.dof_vector = discrete_solution.end_values
    calculator.gravitational_acceleration = 1
    calculator.bottom_topography = np.array([0, 1])
    calculator.volume_space = volume_space
    calculator.intermediate_velocities = velocities
    calculator.source_term_discretization = source_term

    calculator.setup(1)

    def test_cell_indices(self):
        self.assertTupleEqual(self.calculator.cell_indices, (0, 1))

    def test_swe_values(self):
        for i in range(2):
            self.assertListEqual(
                list(self.calculator.swe_values[i]),
                list(self.discrete_solution.end_values[i]),
            )

    def test_heights(self):
        self.assertTupleEqual(self.calculator.heights, (1, 4))

    def test_dischares(self):
        self.assertTupleEqual(self.calculator.discharges, (-1, 4))

    def test_fluxes(self):
        expected_fluxes = [[-1, 1.5], [4, 12]]
        for i in range(2):
            self.assertListEqual(list(self.calculator.fluxes[i]), expected_fluxes[i])

    def test_wave_velocities(self):
        self.assertTupleEqual(self.calculator.wave_velocities, (-2, 3))

    def test_cell_fluxes(self):
        expected_cell_fluxes = [[-3.8, 0.7], [-3.8, -1.8]]

        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    self.calculator.cell_fluxes[i][j], expected_cell_fluxes[i][j]
                )

    def test_topography_step(self):
        self.assertEqual(self.calculator.topography_step, 1)


class TestSWEGodunovFlux(TestCase):
    discrete_solution = DiscreteSolutionObservable(
        0, np.array([[1, 1], [0, 0], [1, -1], [2, 0]])
    )

    velocities = SWEIntermediateVelocities(discrete_solution)
    velocities.volume_space = VOLUME_SPACE
    velocities.gravitational_acceleration = 1
    velocities.update()

    source_term = NaturalSourceTermDiscretization()
    source_term.step_length = 0.5

    calculator = GodunovCellFluxesCalculator()
    calculator.dof_vector = discrete_solution.end_values
    calculator.gravitational_acceleration = 1
    calculator.bottom_topography = np.array([1, 1, 1, 1])
    calculator.volume_space = VOLUME_SPACE
    calculator.intermediate_velocities = velocities
    calculator.source_term_discretization = source_term

    numerical_flux = SWEGodunovNumericalFlux()
    numerical_flux.volume_space = VOLUME_SPACE
    numerical_flux.cell_flux_calculator = calculator

    expected_left_flux = np.array(
        [[0.0, 2.0], [-1.41421356, 1.70710678], [-1.41421356, 0.87867966], [0.0, 2.0]]
    )
    expected_right_flux = np.array(
        [[0.0, 2.0], [0.0, 2.0], [-1.41421356, 1.70710678], [-1.41421356, 0.87867966]]
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
