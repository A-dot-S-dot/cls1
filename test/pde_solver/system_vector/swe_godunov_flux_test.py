from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import numpy as np
from pde_solver.system_vector import SWEGodunovNumericalFlux


class TestSWEGodunovFlux(TestCase):
    numerical_flux = SWEGodunovNumericalFlux()
    numerical_flux.volume_space = VOLUME_SPACE
    numerical_flux.bottom_topography = np.array([1, 1, 1, 1])
    numerical_flux.gravitational_acceleration = 1
    test_dofs = np.array([[1, 0, 1, 2], [1, 0, -1, 0]])
    expected_flux = np.array(
        [
            [0.24264069, 1.0, 1.24264069, -2.48528137],
            [-0.53553391, 1.5, -0.96446609, 0.0],
        ]
    )

    def test_numerical_flux(self):
        numerical_flux = self.numerical_flux(self.test_dofs)
        for i in range(len(numerical_flux)):
            for j in range(2):
                self.assertAlmostEqual(
                    numerical_flux[i, j], self.expected_flux[i, j], msg=f"index={i}"
                )
