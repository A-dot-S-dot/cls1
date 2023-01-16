from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, spmatrix
from scipy.sparse.linalg import SuperLU, splu, spsolve
from tqdm import tqdm

from base.discretization.finite_element import (
    LagrangeSpace,
    LocalLagrangeBasis,
)
from base.quadrature import LocalElementQuadrature
from base.quadrature.local import LocalElementQuadrature
from base.time_stepping import TimeStepping


class VectorEntryCalculator(ABC):
    """Class for calculating a vector entry using local to global principles."""

    @abstractmethod
    def __call__(self, cell_index: int, local_index: int) -> float:
        ...


class SystemMatrixEntryCalculator(ABC):
    """Object for calculating a matrix entry."""

    @abstractmethod
    def __call__(
        self, cell_index: int, local_index_1: int, local_index_2: int
    ) -> float:
        ...


class QuadratureBasedVectorEntryCalculator(VectorEntryCalculator):
    """An Entry Calculator which entries are integrals.

    It's call method must be implemented by subclasses.
    """

    _local_quadrature: LocalElementQuadrature

    def __init__(self, element_space: LagrangeSpace, quadrature_degree: int):
        self._element_space = element_space
        self._local_quadrature = LocalElementQuadrature(quadrature_degree)


class QuadratureBasedMatrixEntryCalculator(SystemMatrixEntryCalculator):
    """An Entry Calculator which entries are integrals.

    It's call method must be implemented by subclasses.
    """

    _local_quadrature: LocalElementQuadrature
    _local_basis: LocalLagrangeBasis

    def __init__(self, polynomial_degree: int, quadrature_degree: int):
        self._local_quadrature = LocalElementQuadrature(quadrature_degree)
        self._local_basis = LocalLagrangeBasis(polynomial_degree)


class SystemVector(ABC):
    @abstractmethod
    def __call__(self, *args) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class SystemMatrix:
    dimension: int

    _inverse: SuperLU
    _csr_values: csr_matrix
    _lil_values: lil_matrix

    def __init__(self, dimension: int):
        self.dimension = dimension

        self._inverse_function = lambda vector: spsolve(self._csr_values, vector)
        self._lil_values = lil_matrix((dimension, dimension))

    def __call__(self) -> spmatrix:
        return self._lil_values

    def assemble(self, *args):
        ...

    def set_values(self, values: np.ndarray):
        self._csr_values = csr_matrix(values)
        self._lil_values = lil_matrix(values)

    def build_inverse(self):
        self._inverse = splu(self._csr_values.tocsc())
        self._inverse_function = lambda vector: self._inverse.solve(vector)

    @property
    def inverse(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._inverse_function

    def update_csr_values(self):
        self._csr_values = csr_matrix(self._lil_values)

    def __getitem__(self, key):
        return self._lil_values[key]

    def __setitem__(self, key, value):
        self._lil_values[key] = value

    def __repr__(self) -> str:
        return self._lil_values.toarray().__repr__()

    def __add__(self, other):
        return self._csr_values + other._csr_values

    def __sub__(self, other):
        return self._csr_values - other._csr_values

    def dot(self, vector: np.ndarray):
        return self._csr_values.dot(vector)

    def multiply_row(self, vector: np.ndarray) -> spmatrix:
        """Multiply each row of matrix with VECTOR elementwise. To be more
        precise we get mij*vj.

        """
        return self().multiply(vector)

    def multiply_column(self, vector: np.ndarray) -> spmatrix:
        """Multiply each row of matrix with VECTOR elementwise. To be more
        precise we get mij*vi.

        """
        return self().multiply(vector.reshape((len(vector), 1)))


class LocallyAssembledVector(SystemVector):
    """Assembles with 'local to global' principles."""

    _element_space: LagrangeSpace
    _entry_calculator: VectorEntryCalculator

    def __init__(
        self,
        element_space: LagrangeSpace,
        entry_calculator: VectorEntryCalculator,
    ):
        self._element_space = element_space
        self._entry_calculator = entry_calculator

    def __call__(self) -> np.ndarray:
        vector = np.zeros(self._element_space.dimension)

        for cell_index in range(len(self._element_space.mesh)):
            for local_index in range(self._element_space.polynomial_degree + 1):
                global_index = self._element_space.global_index(cell_index, local_index)

                vector[global_index] += self._entry_calculator(cell_index, local_index)

        return vector


class LocallyAssembledSystemMatrix(SystemMatrix):
    """The matrix is filled using from local to global principles.

    We loop through each element of the mesh and its local indices, calculate
    something for that and add it to the related entry of the global matrix.

    """

    _element_space: LagrangeSpace
    _entry_calculator: SystemMatrixEntryCalculator
    _dimension: int

    def __init__(
        self,
        element_space: LagrangeSpace,
        entry_calculator: SystemMatrixEntryCalculator,
    ):
        SystemMatrix.__init__(self, element_space.dimension)
        self._element_space = element_space
        self._entry_calculator = entry_calculator
        self.assemble()

    def assemble(self):
        for cell_index in range(len(self._element_space.mesh)):
            for local_index_1 in range(self._element_space.polynomial_degree + 1):
                for local_index_2 in range(self._element_space.polynomial_degree + 1):
                    global_index_1 = self._element_space.global_index(
                        cell_index, local_index_1
                    )
                    global_index_2 = self._element_space.global_index(
                        cell_index, local_index_2
                    )

                    self[global_index_1, global_index_2] += self._entry_calculator(
                        cell_index, local_index_1, local_index_2
                    )

        self.update_csr_values()
