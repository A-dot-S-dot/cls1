from unittest import TestCase

import numpy as np
from system.vector.group_finite_element_approximation import (
    GroupFiniteElementApproximation,
)

from ...test_helper import LINEAR_DOF_VECTOR


class TestGroupFiniteElementApproximationBuilder(TestCase):
    dof_vector = LINEAR_DOF_VECTOR
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]

    def test_approximation_linear_flux(self):
        flux = lambda x: x
        self._test_approximation(flux)

    def _test_approximation(self, flux):
        approximation = GroupFiniteElementApproximation(self.dof_vector, flux)
        for dofs in self.test_dofs:
            self.dof_vector.dofs = dofs

            for i, dof in enumerate(dofs):
                self.assertAlmostEqual(
                    approximation[i], flux(dof), msg=f"index={i}, dofs={dofs}"
                )

    def test_approximation_burgers_flux(self):
        flux = lambda x: 1 / 2 * x**2
        self._test_approximation(flux)
