from test.test_helper import LINEAR_LAGRANGE_SPACE, QUADRATIC_LAGRANGE_SPACE
from test.test_helper.lagrange_basis_elements import *
from typing import Callable, Sequence
from unittest import TestCase

import numpy as np
from core import AffineTransformation, FastFunction, LocalElementQuadrature, Mesh

from lib import BasisGradientL2Product, BasisL2Product

ScalarFunction = Callable[[float], float]


class FastMapping(FastFunction):
    _function: ScalarFunction
    _derivative: ScalarFunction
    _mesh: Mesh
    _local_points: Sequence[float]

    _affine_transformation: AffineTransformation

    def __init__(
        self,
        function: ScalarFunction,
        derivative: ScalarFunction,
        mesh: Mesh,
        local_points: Sequence[float],
    ):
        self._function = function
        self._derivative = derivative
        self._mesh = mesh
        self._local_points = local_points

        self._affine_transformation = AffineTransformation()

    def __call__(self, cell_index: int, local_index: int) -> float:
        cell = self._mesh[cell_index]
        local_point = self._local_points[local_index]

        return self._function(self._affine_transformation(local_point, cell))

    def derivative(self, simplex_index: int, local_index: int) -> float:
        simplex = self._mesh[simplex_index]
        local_point = self._local_points[local_index]

        return self._derivative(self._affine_transformation(local_point, simplex))


class TestLinearBasisL2Product(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    quadrature_degree = 2
    quadrature = LocalElementQuadrature(quadrature_degree)
    test_functions = [lambda _: 1, lambda x: x, lambda x: x**2 - 2 * x]
    test_fast_functions = [
        FastMapping(lambda _: 1, lambda _: 0, element_space.mesh, quadrature.nodes),
        FastMapping(lambda x: x, lambda _: 1, element_space.mesh, quadrature.nodes),
        FastMapping(
            lambda x: x**2 - 2 * x,
            lambda x: 2 * x - 2,
            element_space.mesh,
            quadrature.nodes,
        ),
    ]
    test_functions_strings = ["1", "x", "x²-2x"]
    basis = basis1
    expected_dofs = [
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.125, 0.0625, 0.125, 0.1875]),
        np.array([-0.14322917, -0.10677083, -0.18489583, -0.23177083]),
    ]
    l2_product_class = BasisL2Product

    def test_product(self):
        for fast_function, functions_string, expected_l2_product in zip(
            self.test_fast_functions,
            self.test_functions_strings,
            self.expected_dofs,
        ):
            l2_product_object = self.l2_product_class(
                fast_function, self.element_space, self.quadrature_degree
            )
            l2_product = l2_product_object()

            for i in range(self.element_space.dimension):
                self.assertAlmostEqual(
                    l2_product[i],
                    expected_l2_product[i],
                    msg=f"entry={i}, f(x)={functions_string}",
                )


class TestLinearBasisGradientL2Product(TestLinearBasisL2Product):
    element_space = LINEAR_LAGRANGE_SPACE
    quadrature_degree = 2
    quadrature = LocalElementQuadrature(quadrature_degree)
    basis = basis1_derivative
    l2_product_class = BasisGradientL2Product
    expected_dofs = [
        np.array([0, 0, 0, 0]),
        np.array([0.75, -0.25, -0.25, -0.25]),
        np.array([-0.75, 0.375, 0.25, 0.125]),
    ]


class TestQuadraticBasisL2Product(TestLinearBasisL2Product):
    element_space = QUADRATIC_LAGRANGE_SPACE
    quadrature_degree = 3
    quadrature = LocalElementQuadrature(quadrature_degree)
    test_fast_functions = [
        FastMapping(lambda _: 1, lambda _: 0, element_space.mesh, quadrature.nodes),
        FastMapping(lambda x: x, lambda _: 1, element_space.mesh, quadrature.nodes),
        FastMapping(
            lambda x: x**2 - 2 * x,
            lambda x: 2 * x - 2,
            element_space.mesh,
            quadrature.nodes,
        ),
    ]
    basis = basis2
    expected_dofs = [
        np.array([1 / 6, 1 / 3, 1 / 6, 1 / 3]),
        np.array([0.08333333, 0.08333333, 0.08333333, 0.25]),
        np.array([-0.0875, -0.14166667, -0.12916667, -0.30833333]),
    ]


class TestQuadraticBasisGradientL2Product(TestLinearBasisGradientL2Product):
    element_space = QUADRATIC_LAGRANGE_SPACE
    quadrature_degree = 3
    quadrature = LocalElementQuadrature(quadrature_degree)
    test_fast_functions = [
        FastMapping(lambda _: 1, lambda _: 0, element_space.mesh, quadrature.nodes),
        FastMapping(lambda x: x, lambda _: 1, element_space.mesh, quadrature.nodes),
        FastMapping(
            lambda x: x**2 - 2 * x,
            lambda x: 2 * x - 2,
            element_space.mesh,
            quadrature.nodes,
        ),
    ]
    basis = basis2_derivative
    expected_dofs = [
        np.array([0, 0, 0, 0]),
        np.array([0.8333333, -1 / 3, -1 / 6, -1 / 3]),
        np.array([-0.83333333, 0.5, 0.16666667, 0.16666667]),
    ]
