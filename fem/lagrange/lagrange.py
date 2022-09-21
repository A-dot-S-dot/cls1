"""Provides Lagrange Finite Elements."""
import numpy as np
from math_type import FunctionRealToReal
from mesh import Interval, Mesh
from mesh.transformation import AffineTransformation

from ..abstracts import FiniteElementSpace
from ..dof_index_mapping import DOFIndexMapping, PeriodicDOFIndexMapping
from .local_lagrange import LocalLagrangeBasis


class LagrangeFiniteElementSpace(FiniteElementSpace):
    _mesh: Mesh
    _polynomial_degree: int
    _local_basis: LocalLagrangeBasis
    _dof_index_mapping: DOFIndexMapping
    _affine_transformation: AffineTransformation
    _basis_nodes: np.ndarray

    def __init__(self, mesh: Mesh, polynomial_degree: int):
        self._mesh = mesh
        self._polynomial_degree = polynomial_degree
        self._local_basis = LocalLagrangeBasis(polynomial_degree)
        self._dof_index_mapping = PeriodicDOFIndexMapping(
            self._mesh, len(self._local_basis)
        )
        self._affine_transformation = AffineTransformation()

        self._build_basis_nodes()

    def _build_basis_nodes(self):
        self._basis_nodes = np.empty(self.dimension)

        for simplex_index, simplex in enumerate(self.mesh):
            for local_index, node in enumerate(self.local_basis.nodes):
                point = self._affine_transformation(node, simplex)
                global_index = self._dof_index_mapping(simplex_index, local_index)

                self._basis_nodes[global_index] = point

        self._adjust_basis_nodes()

    def _adjust_basis_nodes(self):
        if isinstance(self._dof_index_mapping, PeriodicDOFIndexMapping):
            self._basis_nodes[0] = self.domain.a
        else:
            raise NotImplementedError

    @property
    def polynomial_degree(self) -> int:
        return self._polynomial_degree

    @property
    def dimension(self) -> int:
        return self._dof_index_mapping.output_dimension

    @property
    def domain(self) -> Interval:
        return self._mesh.domain

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def indices_per_simplex(self) -> int:
        return self.polynomial_degree + 1

    @property
    def local_basis(self) -> LocalLagrangeBasis:
        return self._local_basis

    @property
    def basis_nodes(self) -> np.ndarray:
        return self._basis_nodes

    def get_global_index(self, simplex_index: int, local_index: int) -> int:
        return self._dof_index_mapping(simplex_index, local_index)

    def get_value(self, point: float, dof_vector: np.ndarray) -> float:
        simplex_index = self._mesh.find_simplex_indices(point)[0]
        return self.get_value_on_simplex(point, dof_vector, simplex_index)

    def get_value_on_simplex(
        self, point: float, dof_vector: np.ndarray, simplex_index: int
    ) -> float:
        simplex = self._mesh[simplex_index]
        value = 0

        for local_index, local_element in enumerate(self.local_basis):
            global_index = self._dof_index_mapping(simplex_index, local_index)
            value += dof_vector[global_index] * local_element(
                self._affine_transformation.inverse(point, simplex)
            )

        return value

    def get_derivative(self, point: float, dof_vector: np.ndarray) -> float:
        simplex_index = self._mesh.find_simplex_indices(point)[0]
        simplex = self._mesh[simplex_index]

        # derivatives at boundary of an simplex are not defined in general
        if simplex.is_in_boundary(point):
            return np.nan
        else:
            return self.get_derivative_on_simplex(point, dof_vector, simplex_index)

    def get_derivative_on_simplex(
        self, point: float, dof_vector: np.ndarray, simplex_index: int
    ) -> float:
        simplex = self._mesh[simplex_index]

        value = 0
        for local_index, local_element in enumerate(self.local_basis):
            global_index = self._dof_index_mapping(simplex_index, local_index)
            local_derivative = np.array(
                local_element.derivative(
                    self._affine_transformation.inverse(point, simplex)
                )
            )
            value += dof_vector[global_index] * local_derivative

        return self._affine_transformation.inverse_derivative(simplex) * value

    def interpolate(self, function: FunctionRealToReal) -> np.ndarray:
        if isinstance(self._dof_index_mapping, PeriodicDOFIndexMapping):
            self._check_continuity_at_periodic_point(function)

        dof_vector = np.array([function(node) for node in self.basis_nodes])

        return dof_vector

    def _check_continuity_at_periodic_point(self, function: FunctionRealToReal):
        eps = 1e-12

        if abs(function(self.mesh.domain.a) - function(self.mesh.domain.b)) > eps:
            raise ValueError(
                f"the given function is not continuous in {self.mesh.domain.a}"
            )
