from itertools import combinations
import numpy as np
from pde_solver.solver_space import LagrangeFiniteElementSpace

from pde_solver.system_matrix import SystemMatrix

from .low_order_cg import LowOrderCGRightHandSide
from .system_vector import SystemVector


class MCLRightHandSide(SystemVector):
    """Right hand side of MCL limiting. To be more
    precise it is defined as following:

        ri = 1/mi*sum(dij*(uj-ui)-(fj-fi)*cij + f*_ij, j!=i)

    where mi denotes lumped mass, dij an artificial diffusion, ui the DOF
    entries, fi flux approximation, cij a discrete gradient and f*_ij a
    corrected flux. The corrected flux is calculated as follows:

        f*_ij = max(fmin_ij, min(fij, fmax_ij)),
        fmax_ij = min(2*dij*umax_i-wij, wji-2*dji*umin_j),
        fmin_ij = max(2*dij*umin_i-wij, wji-2*dji*umax_j),
        wij = dij*(uj+ui)+(fi-fj)*cij,
        fij = mij*(dudt_i-dudt_j)+dij*(ui-uj) (target flux)

    where mij denotes the mass matrix and dudt_i the right hand side of low
    order cg.

    """

    mass: SystemMatrix
    local_minimum: SystemVector
    local_maximum: SystemVector

    _element_space: LagrangeFiniteElementSpace
    _low_cg_right_hand_side: LowOrderCGRightHandSide
    _lumped_mass: SystemVector
    _artificial_diffusion: SystemMatrix
    _discrete_gradient: SystemMatrix
    _flux_approximation: SystemVector

    def __init__(self, element_space: LagrangeFiniteElementSpace):
        self._element_space = element_space

    @property
    def low_cg_right_hand_side(self):
        ...

    @low_cg_right_hand_side.setter
    def low_cg_right_hand_side(self, low_cg: LowOrderCGRightHandSide):
        self._low_cg_right_hand_side = low_cg
        self._lumped_mass = low_cg.lumped_mass
        self._artificial_diffusion = low_cg.artificial_diffusion
        self._discrete_gradient = low_cg.discrete_gradient
        self._flux_approximation = low_cg.flux_approximation

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        corrected_flux = np.zeros(len(dof_vector))
        flux_approximation = self._flux_approximation(dof_vector)
        right_hand_side = self._low_cg_right_hand_side(dof_vector)
        local_maximum = self.local_maximum(dof_vector)
        local_minimum = self.local_minimum(dof_vector)

        for element_index in range(len(self._element_space.mesh)):
            for (i_local, j_local) in combinations(
                range(self._element_space.polynomial_degree + 1), 2
            ):
                i = self._element_space.global_index(element_index, i_local)
                j = self._element_space.global_index(element_index, j_local)

                mij = self.mass[i, j]
                dij = self._artificial_diffusion[i, j]
                cij = self._discrete_gradient[i, j]

                ui = dof_vector[i]
                uj = dof_vector[j]
                fi = flux_approximation[i]
                fj = flux_approximation[j]
                dudt_i = right_hand_side[i]
                dudt_j = right_hand_side[j]
                umax_i = local_maximum[i]
                umax_j = local_maximum[j]
                umin_i = local_minimum[i]
                umin_j = local_minimum[j]

                fij = mij * (dudt_i - dudt_j) + dij * (ui - uj)
                wij = dij * (uj + ui) + (fi - fj) * cij

                fmin_ij = max(2 * dij * umin_i - wij, wij - 2 * dij * umax_j)
                fmax_ij = min(2 * dij * umax_i - wij, wij - 2 * dij * umin_j)
                fstar_ij = max(fmin_ij, min(fmax_ij, fij))

                corrected_flux[i] += fstar_ij
                corrected_flux[j] -= fstar_ij

        return right_hand_side + corrected_flux / self._lumped_mass()
