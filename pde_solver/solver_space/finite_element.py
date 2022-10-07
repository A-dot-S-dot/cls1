import numpy as np
from pde_solver.discretization.local_lagrange import LOCAL_LAGRANGE_BASIS
from pde_solver.index_mapping.finite_element import (
    DOFNeighbourIndicesMapping,
    GlobalIndexMapping,
)
from pde_solver.mesh import Mesh, AffineTransformation

from .solver_space import SolverSpace


class LagrangeFiniteElementSpace(SolverSpace):
    mesh: Mesh
    polynomial_degree: int
    global_index: GlobalIndexMapping
    dof_neighbours: DOFNeighbourIndicesMapping
    basis_nodes: np.ndarray

    def __init__(self, mesh: Mesh, polynomial_degree: int):
        self.mesh = mesh
        self.polynomial_degree = polynomial_degree
        self.global_index = GlobalIndexMapping(mesh, polynomial_degree)
        self.dof_neighbours = DOFNeighbourIndicesMapping(
            mesh, polynomial_degree, self.dimension
        )
        self._build_basis_nodes()

    def _build_basis_nodes(self):
        self.basis_nodes = np.empty(self.dimension)
        local_basis = LOCAL_LAGRANGE_BASIS[self.polynomial_degree]
        affine_transformation = AffineTransformation()

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
