from test.test_helper.lagrange_basis_elements import *
from unittest import TestCase

import numpy as np
from pde_solver.discretization.finite_element import (
    AnchorNodesFastFiniteElement,
    FastLocalElement,
    QuadratureFastFiniteElement,
    FastFunction,
)
from pde_solver.discretization.local_lagrange import LocalLagrangeBasis
from pde_solver.mesh import AffineTransformation, Interval, UniformMesh
from pde_solver.quadrature import LocalElementQuadrature
from pde_solver.solver_space import LagrangeFiniteElementSpace


class TestFastFunction(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    local_quadrature = LocalElementQuadrature(2)
    nodes = local_quadrature.nodes
    test_functions = [lambda x: x, lambda x: np.sin(x)]
    test_derivatives = [lambda _: 1, lambda x: np.cos(x)]
    test_function_strings = ["x", "sin(x)"]
    affine_transformation = AffineTransformation()

    def test_fast_function_values(self):
        for function, derivative, f_string in zip(
            self.test_functions, self.test_derivatives, self.test_function_strings
        ):
            fast_function = FastFunction(function, derivative, self.mesh, self.nodes)

            for simplex_index, simplex in enumerate(self.mesh):
                for node_index, node in enumerate(self.nodes):
                    point = self.affine_transformation(node, simplex)
                    self.assertAlmostEqual(
                        fast_function(simplex_index, node_index),
                        function(point),
                        msg=f"K={simplex}, x={point}, f(x)={f_string}",
                    )

    def test_fast_function_derivatives(self):
        for function, derivative, f_string in zip(
            self.test_functions, self.test_derivatives, self.test_function_strings
        ):
            fast_function = FastFunction(function, derivative, self.mesh, self.nodes)

            for simplex_index, simplex in enumerate(self.mesh):
                for node_index, node in enumerate(self.nodes):
                    point = self.affine_transformation(node, simplex)
                    self.assertAlmostEqual(
                        fast_function.derivative(simplex_index, node_index),
                        derivative(point),
                        msg=f"K={simplex}, x={point}, f(x)={f_string}",
                    )


class TestFastLocalElement(TestCase):
    local_basis = LocalLagrangeBasis(1)
    nodes = [0, 0.5, 1]
    fast_element = FastLocalElement(local_basis[0])
    expected_values = [1, 0.5, 0]
    expected_derivatives = [-1, -1, -1]

    def test_fast_local_element_values(self):
        self.fast_element.set_values(*self.nodes)

        for node_index, node in enumerate(self.nodes):
            self.assertAlmostEqual(
                self.fast_element(node_index),
                self.expected_values[node_index],
                msg=f"node={node}",
            )

    def test_fast_local_element_derivatives(self):
        self.fast_element.set_derivatives(*self.nodes)

        for node_index, node in enumerate(self.nodes):
            self.assertAlmostEqual(
                self.fast_element.derivative(node_index),
                self.expected_derivatives[node_index],
                msg=f"node={node}",
            )


class TestLinearQuadratureFastFiniteElement(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 4)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    local_quadrature = LocalElementQuadrature(2)
    fast_element = QuadratureFastFiniteElement(element_space, local_quadrature)
    nodes = local_quadrature.nodes
    test_dof = np.array([1, 0, 0, 0])
    test_finite_element = [phi1_01, phi1_00]
    test_finite_element_derivative = [phi1_01.deriv(), phi1_00.deriv()]
    affine_transformation = AffineTransformation()

    def test_fast_element_values(self):
        self.fast_element.dof_vector = self.test_dof
        self.fast_element.set_values()

        for index, simplex in enumerate([self.mesh[0], self.mesh[-1]]):
            simplex_index = self.mesh.index(simplex)
            for node_index, node in enumerate(self.nodes):
                self.assertAlmostEqual(
                    self.fast_element(simplex_index, node_index),
                    self.test_finite_element[index](
                        self.affine_transformation(node, simplex)
                    ),
                )

    def test_fast_element_derivative(self):
        self.fast_element.dof_vector = self.test_dof
        self.fast_element.set_derivatives()

        for index, simplex in enumerate([self.mesh[0], self.mesh[-1]]):
            simplex_index = self.mesh.index(simplex)
            for node_index, node in enumerate(self.nodes):
                self.assertAlmostEqual(
                    self.fast_element.derivative(simplex_index, node_index),
                    self.test_finite_element_derivative[index](
                        self.affine_transformation(node, simplex)
                    ),
                )


class TestQuadraticQuadratureFastFiniteElement(TestLinearQuadratureFastFiniteElement):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    local_quadrature = LocalElementQuadrature(3)
    fast_element = QuadratureFastFiniteElement(element_space, local_quadrature)
    nodes = local_quadrature.nodes
    test_dof = np.array([1, 0, 0, 0])
    test_finite_element = [phi2_00, phi2_01]
    test_finite_element_derivative = [phi2_00.deriv(), phi2_01.deriv()]


class TestLinearAnchorNodesFastFiniteElement(TestLinearQuadratureFastFiniteElement):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 4)
    element_space = LagrangeFiniteElementSpace(mesh, 1)
    local_quadrature = LocalElementQuadrature(2)
    fast_element = AnchorNodesFastFiniteElement(element_space)
    nodes = fast_element._anchor_nodes
    test_dof = np.array([1, 0, 0, 0])
    test_finite_element = [phi1_01, phi1_00]
    test_finite_element_derivative = [phi1_01.deriv(), phi1_00.deriv()]


class TestQuadraticAnchorNodesFastFiniteElement(TestLinearQuadratureFastFiniteElement):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    local_quadrature = LocalElementQuadrature(2)
    fast_element = AnchorNodesFastFiniteElement(element_space)
    nodes = fast_element._anchor_nodes
    test_dof = np.array([1, 0, 0, 0])
    test_finite_element = [phi2_00, phi2_01]
    test_finite_element_derivative = [phi2_00.deriv(), phi2_01.deriv()]
