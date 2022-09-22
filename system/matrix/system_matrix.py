import numpy as np
from fem import FiniteElementSpace
from math_type import FunctionRealDToRealD
from scipy.sparse import csr_matrix, lil_matrix, spmatrix
from scipy.sparse.linalg import SuperLU, splu, spsolve

from .entry_calculator import SystemMatrixEntryCalculator


class SystemMatrix:
    _element_space: FiniteElementSpace
    _inverse: SuperLU
    _csr_values: csr_matrix
    _lil_values: lil_matrix

    def __init__(self, element_space: FiniteElementSpace):
        self._element_space = element_space
        self._inverse_function = lambda vector: spsolve(self._csr_values, vector)
        self._lil_values = lil_matrix(
            (element_space.dimension, element_space.dimension)
        )

    def set_values(self, values: np.ndarray):
        self._csr_values = csr_matrix(values)
        self._lil_values = lil_matrix(values)

    def assemble(self):
        self.update_values()

    def build_inverse(self):
        self._inverse = splu(self._csr_values.tocsc())
        self._inverse_function = lambda vector: self._inverse.solve(vector)

    @property
    def inverse(self) -> FunctionRealDToRealD:
        return self._inverse_function

    @property
    def dimension(self) -> int:
        return self.element_space.dimension

    @property
    def element_space(self) -> FiniteElementSpace:
        return self._element_space

    @property
    def values(self) -> spmatrix:
        return self._lil_values

    def update_values(self):
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


class LocallyAssembledSystemMatrix(SystemMatrix):
    """The matrix is filled using from local to global principles.

    We loop through each element of the mesh and its local indices, calculate
    something for that and add it to the related entry of the global matrix.

    """

    _element_space: FiniteElementSpace
    _entry_calculator: SystemMatrixEntryCalculator
    _dimension: int

    def __init__(
        self,
        element_space: FiniteElementSpace,
        entry_calculator: SystemMatrixEntryCalculator,
    ):
        SystemMatrix.__init__(self, element_space)
        self._entry_calculator = entry_calculator

    def assemble(self):
        for simplex_index in range(len(self.element_space.mesh)):
            for local_index_1 in range(self.element_space.indices_per_simplex):
                for local_index_2 in range(self.element_space.indices_per_simplex):
                    global_index_1 = self.element_space.get_global_index(
                        simplex_index, local_index_1
                    )
                    global_index_2 = self.element_space.get_global_index(
                        simplex_index, local_index_2
                    )

                    self[global_index_1, global_index_2] += self._entry_calculator(
                        simplex_index, local_index_1, local_index_2
                    )

        self.update_values()
