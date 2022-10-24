from test.test_helper import VOLUME_SPACE
from unittest import TestCase

import numpy as np
from pde_solver.system_flux import NetworkApproximatedFlatBottomSubgridFlux


class TestFlatBottomCoarseFlux(TestCase):
    volume_space = VOLUME_SPACE
    flux = NetworkApproximatedFlatBottomSubgridFlux(
        "test/test_helper/test_train.csv", "test/test_helper/test_subgrid_network.pth"
    )
    test_dof = np.array([[1, 1], [0, 0], [1, -1], [2, 0]])
    expected_left_flux = np.array(
        [
            [-2.0345347, 9.270818],
            [-1.6421734, 7.00718],
            [2.159339, -9.792055],
            [0.21776798, -9.566265],
        ]
    )

    expected_right_flux = np.array(
        [
            [-1.6421734, 7.00718],
            [2.159339, -9.792055],
            [0.21776798, -9.566265],
            [-2.0345347, 9.270818],
        ],
    )

    def test(self):
        left_flux, right_flux = self.flux(self.test_dof)
        for i in range(4):
            for j in range(2):
                self.assertAlmostEqual(
                    left_flux[i, j], self.expected_left_flux[i, j], places=6
                )
                self.assertAlmostEqual(
                    right_flux[i, j], self.expected_right_flux[i, j], places=6
                )
