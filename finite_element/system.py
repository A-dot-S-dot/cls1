from core.system import *

from .space import *


class QuadratureBasedMatrixEntryCalculator(SystemMatrixEntryCalculator):
    """An Entry Calculator which entries are integrals.

    It's call method must be implemented by subclasses.
    """

    _local_quadrature: LocalElementQuadrature
    _local_basis: LocalLagrangeBasis

    def __init__(self, polynomial_degree: int, quadrature_degree: int):
        self._local_quadrature = LocalElementQuadrature(quadrature_degree)
        self._local_basis = LocalLagrangeBasis(polynomial_degree)


class LocallyAssembledVector:
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
