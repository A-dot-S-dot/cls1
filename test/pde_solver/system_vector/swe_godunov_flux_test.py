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
    expected_left_flux = np.array(
        [[1.0, 0.0, -1.24264069, 1.24264069], [1.5, 0.0, 0.96446609, 0.96446609]]
    )
    expected_right_flux = np.array(
        [[1.24264069, 1.0, 0.0, -1.24264069], [0.96446609, 1.5, 0.0, 0.96446609]]
    )

    def test_numerical_flux(self):
        left_flux, right_flux = self.numerical_flux(self.test_dofs)

        for i in range(2):
            for j in range(VOLUME_SPACE.dimension):
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
