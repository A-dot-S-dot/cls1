from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, spmatrix
from scipy.sparse.linalg import SuperLU, splu, spsolve


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
