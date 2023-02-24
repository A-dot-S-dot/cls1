from typing import Callable, Iterator, List, Sequence

import numpy as np
import core
from scipy.interpolate import lagrange

from .index_mapping import NeighbourIndicesMapping, GlobalIndexMapping


class LocalLagrangeElement:
    """Finite element basis element on the standard simplex."""

    _call_method: Callable[[float], float]
    _derivative: Callable[[float], float]

    def __init__(
        self,
        call_method: Callable[[float], float],
        derivative: Callable[[float], float],
    ):
        self._call_method = call_method
        self._derivative = derivative

    def __call__(self, point: float) -> float:
        return self._call_method(point)

    def derivative(self, point: float) -> float:
        return self._derivative(point)


class LocalLagrangeBasis:
    """Basis of finite elements on the standard simplex [0,1]. Each local basis
    elements can be identified with a node.

    """

    polynomial_degree: int
    nodes: List[float]

    _basis_elements: List[LocalLagrangeElement]

    def __init__(self, polynomial_degree: int):
        self.polynomial_degree = polynomial_degree
        self._build_nodes()
        self._build_elements()

    def _build_nodes(self):
        N = self.polynomial_degree
        self.nodes = [i / N for i in range(N + 1)]

    def _build_elements(self):
        self._basis_elements = [
            self._create_local_finite_element(i) for i in range(len(self.nodes))
        ]

    def _create_local_finite_element(self, index: int) -> LocalLagrangeElement:
        local_element_dof_vector = self._build_unit_vector(index)

        interpolation = lagrange(self.nodes, local_element_dof_vector)
        interpolation_derivative = interpolation.deriv()

        return LocalLagrangeElement(interpolation, interpolation_derivative)

    def _build_unit_vector(self, index: int) -> np.ndarray:
        unit_vector = np.zeros(len(self.nodes))
        unit_vector[index] = 1

        return unit_vector

    def __len__(self) -> int:
        return len(self._basis_elements)

    def __iter__(self) -> Iterator[LocalLagrangeElement]:
        return iter(self._basis_elements)

    def __getitem__(self, node_index: int) -> LocalLagrangeElement:
        return self._basis_elements[node_index]

    def get_element_at_node(self, node) -> LocalLagrangeElement:
        return self._basis_elements[self.nodes.index(node)]


class LagrangeSpace(core.SolverSpace[float]):
    mesh: core.Mesh
    polynomial_degree: int
    global_index: GlobalIndexMapping
    dof_neighbours: NeighbourIndicesMapping
    basis_nodes: np.ndarray

    def __init__(self, mesh: core.Mesh, polynomial_degree: int):
        self.mesh = mesh
        self.polynomial_degree = polynomial_degree
        self.global_index = GlobalIndexMapping(mesh, polynomial_degree, periodic=True)
        self.dof_neighbours = NeighbourIndicesMapping(self.global_index)
        self._build_basis_nodes()

    def _build_basis_nodes(self):
        self.basis_nodes = np.empty(self.dimension)
        local_basis = LocalLagrangeBasis(self.polynomial_degree)
        affine_transformation = core.AffineTransformation()

        for cell_index, cell in enumerate(self.mesh):
            for local_index, node in enumerate(local_basis.nodes):
                point = affine_transformation(node, cell)
                global_index = self.global_index(cell_index, local_index)

                self.basis_nodes[global_index] = point

        self._adjust_basis_nodes()

    def _adjust_basis_nodes(self):
        if self.global_index.periodic:
            self.basis_nodes[0] = self.mesh.domain.a
        else:
            raise NotImplementedError

    @property
    def dimension(self):
        # considered periodic boundary
        return len(self.mesh) * self.polynomial_degree

    @property
    def grid(self) -> np.ndarray:
        return self.basis_nodes

    def element(self, dof_vector: np.ndarray) -> core.CellDependentFunction[float]:
        return LagrangeFiniteElement(self, dof_vector)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + f"(mesh={self.mesh}, p={self.polynomial_degree})"
        )


class LagrangeFiniteElement(core.CellDependentFunction[float]):
    """Finite element which is defined by coefficients each belonging to a basis
    element of the finite element space."""

    _element_space: LagrangeSpace
    _mesh: core.Mesh
    _dof_vector: np.ndarray
    _local_basis: LocalLagrangeBasis
    _affine_transformation: core.AffineTransformation

    def __init__(
        self,
        element_space: LagrangeSpace,
        dof_vector: np.ndarray,
    ):
        self._element_space = element_space
        self._mesh = element_space.mesh
        self._dof_vector = dof_vector
        self._local_basis = LocalLagrangeBasis(element_space.polynomial_degree)
        self._affine_transformation = core.AffineTransformation()

    def __call__(self, cell_index: int, point: float) -> float:
        cell = self._mesh[cell_index]
        value = 0

        for local_index, local_element in enumerate(self._local_basis):
            global_index = self._element_space.global_index(cell_index, local_index)
            value += self._dof_vector[global_index] * local_element(
                self._affine_transformation.inverse(point, cell)
            )

        return value

    def derivative(self, cell_index: int, point: float) -> float:
        cell = self._mesh[cell_index]
        value = 0

        for local_index, local_element in enumerate(self._local_basis):
            global_index = self._element_space.global_index(cell_index, local_index)
            local_derivative = np.array(
                local_element.derivative(
                    self._affine_transformation.inverse(point, cell)
                )
            )
            value += self._dof_vector[global_index] * local_derivative

        return self._affine_transformation.inverse_derivative(cell) * value


class FastLocalElement:
    """This element is made for better performance while calculating values and
    derivatives of local Lagrange elements.

    Invoke 'set_values' and 'set_derivatives' for adding values and derivatives.

    The desired value can be accessed via 'value' or 'derivative'.

    """

    _local_element: LocalLagrangeElement
    _values: Sequence[float]
    _derivatives: Sequence[float]

    def __init__(self, local_element: LocalLagrangeElement):
        self._local_element = local_element

    def set_values(self, *local_points: float):
        self._values = [self._local_element(point) for point in local_points]

    def set_derivatives(self, *local_points: float):
        self._derivatives = [
            self._local_element.derivative(point) for point in local_points
        ]

    def __call__(self, local_index: int) -> float:
        return self._values[local_index]

    def derivative(self, local_index: int) -> float:
        return self._derivatives[local_index]


class FastFiniteElement(core.FastFunction):
    """This element is made for better performance while calculating values and
    derivatives of Lagrange elements.

    Essentially it calculates finite elements on all points which are related to
    certain local points on the standard simplex. This is done by calculating
    the values (and if needed derivatives) in the preprocessing step by invoking
    'set_values' and 'set_derivatives'.

    The desired value can be accessed via 'value_on_simplex' or
    'derivative_on_simplex'.

    """

    dof_vector: np.ndarray
    local_fast_basis: Sequence[FastLocalElement]

    _element_space: LagrangeSpace
    _affine_transformation: core.AffineTransformation

    def __init__(self, element_space: LagrangeSpace):
        self._element_space = element_space
        self._affine_transformation = core.AffineTransformation()
        self._build_fast_basis()

    def _build_fast_basis(self):
        local_basis = LocalLagrangeBasis(self._element_space.polynomial_degree)
        self.local_fast_basis = [
            FastLocalElement(basis_element) for basis_element in local_basis
        ]

    def set_values(self, *local_points: float):
        for element in self.local_fast_basis:
            element.set_values(*local_points)

    def set_derivatives(self, *local_points: float):
        for element in self.local_fast_basis:
            element.set_derivatives(*local_points)

    def __call__(self, cell_index: int, local_index: int) -> float:
        local_dof_vector = self._build_local_dof_vector(cell_index)
        local_basis_elements_values = [
            element(local_index) for element in self.local_fast_basis
        ]
        return np.dot(local_dof_vector, local_basis_elements_values)

    def _build_local_dof_vector(self, cell_index: int) -> np.ndarray:
        local_dof_vector = np.zeros(self._element_space.polynomial_degree + 1)

        for local_index in range(self._element_space.polynomial_degree + 1):
            global_index = self._element_space.global_index(cell_index, local_index)
            local_dof_vector[local_index] = self.dof_vector[global_index]

        return local_dof_vector

    def derivative(self, cell_index: int, local_index: int) -> float:
        local_dof_vector = self._build_local_dof_vector(cell_index)
        local_basis_elements_derivatives = [
            element.derivative(local_index) for element in self.local_fast_basis
        ]

        cell = self._element_space.mesh[cell_index]
        distortion = self._affine_transformation.inverse_derivative(cell)

        return distortion * np.dot(local_dof_vector, local_basis_elements_derivatives)


class QuadratureFastElement(FastFiniteElement):
    """Element for fast calculation of values and derivatives in quadrature
    nodes.

    """

    _quadrature_nodes: Sequence[float]

    def __init__(
        self,
        element_space: LagrangeSpace,
        local_quadrature: core.LocalElementQuadrature,
    ):
        FastFiniteElement.__init__(self, element_space)

        self._quadrature_nodes = local_quadrature.nodes

    def set_values(self):
        super().set_values(*self._quadrature_nodes)

    def set_derivatives(self):
        super().set_derivatives(*self._quadrature_nodes)


class AnchorNodesFastFiniteElement(FastFiniteElement):
    """Element for fast calculation of values and derivatives in anchor
    nodes of local Lagrange elements.

    """

    _anchor_nodes: Sequence[float]

    def __init__(self, element_space: LagrangeSpace):
        FastFiniteElement.__init__(self, element_space)

        self._anchor_nodes = LocalLagrangeBasis(element_space.polynomial_degree).nodes

    def set_values(self):
        super().set_values(*self._anchor_nodes)

    def set_derivatives(self):
        super().set_derivatives(*self._anchor_nodes)


def get_finite_element_solution(
    benchmark: core.Benchmark,
    mesh_size: int,
    polynomial_degree: int,
    save_history=False,
) -> core.DiscreteSolution[LagrangeSpace]:
    mesh = core.UniformMesh(benchmark.domain, mesh_size)
    space = LagrangeSpace(mesh, polynomial_degree)
    interpolator = core.NodeValuesInterpolator(*space.basis_nodes)
    solution_type = (
        core.DiscreteSolutionWithHistory if save_history else core.DiscreteSolution
    )

    return solution_type(
        interpolator.interpolate(benchmark.initial_data),
        start_time=benchmark.start_time,
        space=space,
    )
