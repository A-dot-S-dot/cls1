from unittest import TestCase

import numpy as np
import shallow_water
from core.finite_volume.boundary import PeriodicBoundaryConditionsApplier
from numpy.testing import assert_almost_equal
from shallow_water.solver import lax_friedrichs


class TestLocalLaxFriedrichNumericalFlux(TestCase):
    dof_vector = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])
    boundary_conditions = PeriodicBoundaryConditionsApplier()
    riemann_solver = shallow_water.RiemannSolver(boundary_conditions, 1.0)
    numerical_flux = lax_friedrichs.LLFNumericalFLux(riemann_solver)

    def test_numerical_flux(self):
        expected_flux_left = np.array(
            [[1.5, 0.75], [1.5, 1.75], [-1.5, 1.75], [-1.5, 0.75]]
        )
        expected_flux_right = -np.array(
            [[1.5, 1.75], [-1.5, 1.75], [-1.5, 0.75], [1.5, 0.75]]
        )
        flux_left, flux_right = self.numerical_flux(0, self.dof_vector)

        assert_almost_equal(flux_left, expected_flux_left)
        assert_almost_equal(flux_right, expected_flux_right)
