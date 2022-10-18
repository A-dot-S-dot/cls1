from pde_solver.discrete_solution.observed_discrete_solution import (
    DiscreteSolutionObservable,
)
from unittest import TestCase
from pde_solver.mesh import Interval, UniformMesh
from pde_solver.solver_space import FiniteVolumeSpace

import numpy as np
from pde_solver.system_flux import SWEIntermediateVelocities


class TestIntermediateVelocities(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    volume_space = FiniteVolumeSpace(mesh)
    discrete_solution = DiscreteSolutionObservable(0, np.array([[1, -1], [4, 4]]))
    velocities = SWEIntermediateVelocities(discrete_solution)
    velocities.volume_space = volume_space
    velocities.gravitational_acceleration = 1
    expected_left_velocities = np.array([-2, -2])
    expected_right_velocities = np.array([3, 3])

    def test_numerical_flux(self):
        self.velocities.update()

        for i in range(self.volume_space.edge_number):
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
