from pde_solver.discretization.finite_element import LagrangeFiniteElementSpace

from .entry_calculator import SystemMatrixEntryCalculator
from .system_matrix import SystemMatrix


class LocallyAssembledSystemMatrix(SystemMatrix):
    """The matrix is filled using from local to global principles.

    We loop through each element of the mesh and its local indices, calculate
    something for that and add it to the related entry of the global matrix.

    """

    _element_space: LagrangeFiniteElementSpace
    _entry_calculator: SystemMatrixEntryCalculator
    _dimension: int

    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
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
