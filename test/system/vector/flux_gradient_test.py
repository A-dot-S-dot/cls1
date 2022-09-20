from unittest import TestCase

import numpy as np
from system.vector.flux_gradient import (
    AdvectionFluxGradient,
    FluxGradient,
    ApproximatedFluxGradient,
)
from system.vector.group_finite_element_approximation import (
    GroupFiniteElementApproximation,
)
from ...test_helper import LINEAR_DOF_VECTOR
from system.matrix.discrete_gradient import DiscreteGradient


class TestFluxGradient(TestCase):
    dof_vector = LINEAR_DOF_VECTOR

    def test_advection_gradient(self):
        flux = lambda x: x
        advection = FluxGradient(self.dof_vector, flux, 2)
        test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
        expected_gradient = (
            -1
            / 2
            * np.array([[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]])
        )
        for dofs in test_dofs:
            self.dof_vector.dofs = dofs
            expected_advection = expected_gradient.dot(dofs)

            for i in range(len(dofs)):
                self.assertAlmostEqual(
                    advection[i], expected_advection[i], msg=f"index={i}, dofs={dofs}"
                )

    def test_burgers_gradient(self):
        flux = lambda x: 1 / 2 * x**2
        test_dofs = np.array([1, 0, 0, 0])
        expected_burgers = np.array([0, 1 / 6, 0, -1 / 6])
        burgers = FluxGradient(self.dof_vector, flux, 2)
        self.dof_vector.dofs = test_dofs

        for i in range(len(test_dofs)):
            self.assertAlmostEqual(burgers[i], expected_burgers[i], msg=f"index={i}")


class TestAdvectionFluxGradient(TestCase):
    dof_vector = LINEAR_DOF_VECTOR
    discrete_gradient = DiscreteGradient(dof_vector.element_space)

    def test_flux_gradient(self):
        advection = AdvectionFluxGradient(self.dof_vector, self.discrete_gradient)
        test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
        expected_gradient = (
            1
            / 2
            * np.array([[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]])
        )
        for dofs in test_dofs:
            self.dof_vector.dofs = dofs
            expected_advection = expected_gradient.dot(dofs)

            for i in range(len(dofs)):
                self.assertAlmostEqual(
                    advection[i], expected_advection[i], msg=f"index={i}, dofs={dofs}"
                )


class TestApproximatedFluxGradient(TestCase):
    dof_vector = LINEAR_DOF_VECTOR
    discrete_gradient = DiscreteGradient(dof_vector.element_space)

    def test_advection_gradient(self):
        flux = lambda x: x
        flux_approximation = GroupFiniteElementApproximation(self.dof_vector, flux)
        advection = ApproximatedFluxGradient(flux_approximation, self.discrete_gradient)
        test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
        expected_gradient = (
            1
            / 2
            * np.array([[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]])
        )
        for dofs in test_dofs:
            self.dof_vector.dofs = dofs
            expected_advection = -expected_gradient.dot(dofs)

            for i in range(len(dofs)):
                self.assertAlmostEqual(
                    advection[i], expected_advection[i], msg=f"index={i}, dofs={dofs}"
                )

    def test_burgers_gradient(self):
        flux = lambda x: 1 / 2 * x**2
        flux_approximation = GroupFiniteElementApproximation(self.dof_vector, flux)
        burgers = ApproximatedFluxGradient(flux_approximation, self.discrete_gradient)
        test_dofs = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])]
        expected_approximations = [
            np.array([0, 1 / 4, 0, -1 / 4]),
            np.array([-1 / 4, 0, 1 / 4, 0]),
        ]

        for dofs, expected_approximation in zip(test_dofs, expected_approximations):
            self.dof_vector.dofs = dofs

            for i in range(len(test_dofs)):
                self.assertAlmostEqual(
                    burgers[i],
                    expected_approximation[i],
                    msg=f"index={i}, dofs={dofs}",
                )
