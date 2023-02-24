from test.test_helper import LINEAR_LAGRANGE_SPACE, discrete_derivative
from test.test_helper.lagrange_basis_elements import *
from unittest import TestCase

import numpy as np
from core.mesh import AffineTransformation, Interval, UniformMesh
from core.quadrature import LocalElementQuadrature

import finite_element


class TestLocalLagrangeBasis(TestCase):
    test_polynomial_degrees = [p + 1 for p in range(8)]

    def test_nodes(self):
        expected_nodes = [
            [0, 1],
            [0, 1 / 2, 1],
            [0, 1 / 3, 2 / 3, 1],
        ]

        for p, expected_basis_nodes in enumerate(expected_nodes):
            basis = finite_element.LocalLagrangeBasis(p + 1)
            for node, expected_node in zip(basis.nodes, expected_basis_nodes):
                self.assertEqual(node, expected_node)

    def test_delta_property(self):
        for p in self.test_polynomial_degrees:
            basis = finite_element.LocalLagrangeBasis(p)
            self._test_delta_property_for_basis(basis)

    def _test_delta_property_for_basis(self, basis: finite_element.LocalLagrangeBasis):
        for i in range(len(basis.nodes)):
            self._compare_basis_element_with_basis(i, basis)

    def _compare_basis_element_with_basis(
        self, basis_index: int, basis: finite_element.LocalLagrangeBasis
    ):
        node_i = basis.nodes[basis_index]
        basis_element = basis.get_element_at_node(node_i)

        for j, node_j in enumerate(basis.nodes):
            self.assertAlmostEqual(
                basis_element(node_j),
                float(node_i == node_j),
                msg=f"p={basis.polynomial_degree}, basis_index={basis_index}, node_index={j}",
            )

    def test_len(self):
        basis = finite_element.LocalLagrangeBasis(2)
        self.assertEqual(len(basis), 3)

    def test_derivative(self):
        for p in range(1, 4):
            basis = finite_element.LocalLagrangeBasis(p)
            self._test_derivative_for_basis(basis)

    def _test_derivative_for_basis(self, basis: finite_element.LocalLagrangeBasis):
        for element_index, element in enumerate(basis):
            self._test_derivative_for_element(element, element_index, basis)

    def _test_derivative_for_element(
        self,
        element: finite_element.LocalLagrangeElement,
        element_index: int,
        basis: finite_element.LocalLagrangeBasis,
    ):
        for x in np.linspace(0, 1):
            self.assertAlmostEqual(
                element.derivative(x),
                discrete_derivative(element, x),
                msg=f"p={basis.polynomial_degree} , element={element_index}, point={x}",
                delta=1e-7,
            )


class TestLinearLagrangeFiniteElementSpace(TestCase):
    polynomial_degree = 1
    mesh_size = 2
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, mesh_size)
    element_space = finite_element.LagrangeSpace(mesh, polynomial_degree)
    expected_basis_nodes = [0, 0.5]

    def test_basis_nodes(self):
        self.assertListEqual(
            list(self.element_space.basis_nodes), self.expected_basis_nodes
        )


class TestQuadraticLagrangeFiniteElementSpace(TestLinearLagrangeFiniteElementSpace):
    polynomial_degree = 2
    mesh_size = 2
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, mesh_size)
    element_space = finite_element.LagrangeSpace(mesh, polynomial_degree)
    expected_basis_nodes = [0, 0.25, 0.5, 0.75]


class TestLagragenFiniteElement(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    element = finite_element.LagrangeFiniteElement(
        element_space, np.array([2, 2, 2, 2])
    )
    points = np.array([0.2, 0.4, 0.6, 0.8])

    def test_element_values(self):
        for i, point in enumerate(self.points):
            self.assertEqual(self.element(i, point), 2)

    def test_element_derivative(self):
        for i, point in enumerate(self.points):
            self.assertEqual(self.element.derivative(i, point), 0)


class TestFastLocalElement(TestCase):
    local_basis = finite_element.LocalLagrangeBasis(1)
    nodes = [0, 0.5, 1]
    fast_element = finite_element.FastLocalElement(local_basis[0])
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
    element_space = finite_element.LagrangeSpace(mesh, 1)
    local_quadrature = LocalElementQuadrature(2)
    fast_element = finite_element.QuadratureFastElement(element_space, local_quadrature)
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
    element_space = finite_element.LagrangeSpace(mesh, 2)
    local_quadrature = LocalElementQuadrature(3)
    fast_element = finite_element.QuadratureFastElement(element_space, local_quadrature)
    nodes = local_quadrature.nodes
    test_dof = np.array([1, 0, 0, 0])
    test_finite_element = [phi2_00, phi2_01]
    test_finite_element_derivative = [phi2_00.deriv(), phi2_01.deriv()]


class TestLinearAnchorNodesFastFiniteElement(TestLinearQuadratureFastFiniteElement):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 4)
    element_space = finite_element.LagrangeSpace(mesh, 1)
    local_quadrature = LocalElementQuadrature(2)
    fast_element = finite_element.AnchorNodesFastFiniteElement(element_space)
    nodes = fast_element._anchor_nodes
    test_dof = np.array([1, 0, 0, 0])
    test_finite_element = [phi1_01, phi1_00]
    test_finite_element_derivative = [phi1_01.deriv(), phi1_00.deriv()]


class TestQuadraticAnchorNodesFastFiniteElement(TestLinearQuadratureFastFiniteElement):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = finite_element.LagrangeSpace(mesh, 2)
    local_quadrature = LocalElementQuadrature(2)
    fast_element = finite_element.AnchorNodesFastFiniteElement(element_space)
    nodes = fast_element._anchor_nodes
    test_dof = np.array([1, 0, 0, 0])
    test_finite_element = [phi2_00, phi2_01]
    test_finite_element_derivative = [phi2_00.deriv(), phi2_01.deriv()]
