from unittest import TestCase
from typing import Tuple

import numpy as np
from pde_solver.discrete_solution.discrete_solution import DiscreteSolution
from pde_solver.system_flux import FlatBottomCoarseFlux, SystemFlux
from pde_solver.time_stepping import TimeStepping
from pde_solver.solver_space import FiniteVolumeSpace
from pde_solver.mesh import Interval, UniformMesh


class TestFlux(SystemFlux):
    def __call__(self, dof_vector) -> Tuple[np.ndarray, np.ndarray]:
        return (np.roll(dof_vector, 1, axis=0), dof_vector)


class TestTimeStepping(TimeStepping):
    start_time = 0
    end_time = 2
    cfl_number = 1

    @property
    def desired_time_step(self) -> float:
        return 0.75


class TestFlatBottomCoarseFlux(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    volume_space = FiniteVolumeSpace(mesh)
    numerical_flux = FlatBottomCoarseFlux()
    numerical_flux.coarse_volume_space = volume_space
    numerical_flux.coarsening_degree = 2
    numerical_flux.fine_solution = DiscreteSolution(
        0, np.array([[0.5, 1], [1, 2], [1.5, 3], [2, 4]])
    )
    numerical_flux.fine_numerical_flux = TestFlux()
    numerical_flux.fine_time_stepping = TestTimeStepping()
    numerical_flux.fine_step_length = 0.25

    time_step = 1
    expected_left_flux = np.array([[4.75, 9.5], [2.75, 5.5]])
    expected_right_flux = np.array([[2.75, 5.5], [4.75, 9.5]])

    def test_numerical_flux(self):
        left_flux, right_flux = self.numerical_flux(self.time_step)

        for i in range(self.volume_space.dimension):
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
