from test.test_helper import LINEAR_LAGRANGE_SPACE
from unittest import TestCase

import numpy as np
from pde_solver.system_matrix import DiscreteGradient
from pde_solver.system_vector import (
    AdvectionFluxGradient,
    ApproximatedFluxGradient,
    FluxGradient,
)


class TestFluxGradient(TestCase):
    def test_advection_gradient(self):
        flux = lambda x: x
        flux_gradient = FluxGradient(LINEAR_LAGRANGE_SPACE, 2, flux)
        test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
        expected_gradient = (
            -1
            / 2
            * np.array([[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]])
        )
        for dofs in test_dofs:
            expected_advection = expected_gradient.dot(dofs)
            result = flux_gradient(dofs)
            for i in range(len(result)):
                self.assertAlmostEqual(
                    result[i], expected_advection[i], msg=f"index={i}"
                )

    def test_burgers_gradient(self):
        flux = lambda x: 1 / 2 * x**2
        test_dofs = np.array([1, 0, 0, 0])
        expected_burgers = [0, 1 / 6, 0, -1 / 6]
        flux_gradient = FluxGradient(LINEAR_LAGRANGE_SPACE, 2, flux)

        result = flux_gradient(test_dofs)
        for i in range(len(result)):
            self.assertAlmostEqual(result[i], expected_burgers[i], msg=f"index={i}")


class TestAdvectionFluxGradient(TestCase):
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)

    def test_flux_gradient(self):
        flux_gradient = AdvectionFluxGradient(self.discrete_gradient)
        test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
        expected_gradient = (
            1
            / 2
            * np.array([[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]])
        )
        for dofs in test_dofs:
            expected_advection = expected_gradient.dot(dofs)
            result = flux_gradient(dofs)
            for i in range(len(result)):
                self.assertAlmostEqual(
                    result[i], expected_advection[i], msg=f"index={i}"
                )


class TestApproximatedFluxGradient(TestCase):
    discrete_gradient = DiscreteGradient(LINEAR_LAGRANGE_SPACE)

    def test_advection_gradient(self):
        flux = lambda x: x
        flux_gradient = ApproximatedFluxGradient(self.discrete_gradient, flux)
        test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]
        expected_gradient = (
            -1
            / 2
            * np.array([[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]])
        )
        for dofs in test_dofs:
            expected_advection = expected_gradient.dot(dofs)
            result = flux_gradient(dofs)
            for i in range(len(result)):
                self.assertAlmostEqual(
                    result[i], expected_advection[i], msg=f"index={i}"
                )

    def test_burgers_gradient(self):
        flux = lambda x: 1 / 2 * x**2
        flux_gradient = ApproximatedFluxGradient(self.discrete_gradient, flux)
        test_dofs = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])]
        expected_approximations = [
            [0, 1 / 4, 0, -1 / 4],
            [-1 / 4, 0, 1 / 4, 0],
        ]

        for dofs, expected_approximation in zip(test_dofs, expected_approximations):
            result = flux_gradient(dofs)
            for i in range(len(result)):
                self.assertAlmostEqual(
                    result[i], expected_approximation[i], msg=f"index={i}"
                )
