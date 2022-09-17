import numpy as np
from fem import FiniteElementSpace
from mesh.uniform import UniformMesh

from .entry_calculator import QuadratureBasedEntryCalculator
from .system_matrix import LocallyAssembledSystemMatrix


class MassEntryCalculator(QuadratureBasedEntryCalculator):
    _determinant_derivative_affine_mapping: float

    def __init__(self, element_space: FiniteElementSpace):
        if not isinstance(element_space.mesh, UniformMesh):
            raise NotImplementedError(
                "Not implemented for different meshes than UniformMesh"
            )

        self._determinant_derivative_affine_mapping = element_space.mesh.step_length
        QuadratureBasedEntryCalculator.__init__(
            self, element_space, element_space.polynomial_degree + 1
        )

    def __call__(
        self, simplex_index: int, local_index_1: int, local_index_2: int
    ) -> float:
        element_1 = self._local_basis[local_index_1]
        element_2 = self._local_basis[local_index_2]

        quadrature_nodes_weights = np.array(
            [element_1(node) * element_2(node) for node in self._local_quadrature.nodes]
        )

        return self._determinant_derivative_affine_mapping * np.dot(
            self._local_quadrature.weights, quadrature_nodes_weights
        )


class MassMatrix(LocallyAssembledSystemMatrix):
    """Mass system matrix. It's entries are phi_i * phi_j, where {phi_i}_i
    denotes the basis of the element space.

    """

    def __init__(self, element_space: FiniteElementSpace):
        entry_calculator = MassEntryCalculator(element_space)
        LocallyAssembledSystemMatrix.__init__(self, element_space, entry_calculator)
        self.assemble()
