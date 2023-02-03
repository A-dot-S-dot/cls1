from unittest import TestCase

import numpy as np
import shallow_water
from numpy.testing import assert_almost_equal
from core import finite_volume


class TestRiemannSolver(TestCase):
    solver = shallow_water.RiemannSolver(
        finite_volume.PeriodicBoundaryConditionsApplier(),
        gravitational_acceleration=1.0,
    )
    vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])

    solver.solve(0.0, vector)

    def test_intermediate_state(self):
        expected_intermediate_state = np.array(
            [
                [1.25, 0.625],
                [0.75, 0.875],
                [0.75, -0.875],
                [1.25, -0.625],
                [1.25, 0.625],
            ]
        )

        assert_almost_equal(self.solver.intermediate_state, expected_intermediate_state)

    def test_flux(self):
        expected_flux = np.array(
            [[1.5, 0.75], [1.5, 1.75], [-1.5, 1.75], [-1.5, 0.75], [1.5, 0.75]]
        )

        assert_almost_equal(self.solver.intermediate_flux, expected_flux)
