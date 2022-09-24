from itertools import combinations
import numpy as np

from system.matrix.system_matrix import SystemMatrix

from .dof_vector import DOFVector
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

    _dof_vector: DOFVector
    _low_cg_right_hand_side: LowOrderCGRightHandSide
    _lumped_mass: SystemVector
    _artificial_diffusion: SystemMatrix
    _discrete_gradient: SystemMatrix
    _flux_approximation: SystemVector

    def __init__(
        self,
        dof_vector: DOFVector,
    ):
        SystemVector.__init__(self, dof_vector.element_space)
        dof_vector.register_observer(self)

        self._dof_vector = dof_vector

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

    def update(self):
        corrected_flux = np.zeros(self.dimension)

        for element_index in range(len(self.element_space.mesh)):
            for (i_local, j_local) in combinations(
                range(self.element_space.indices_per_simplex), 2
            ):
                i = self.element_space.get_global_index(element_index, i_local)
                j = self.element_space.get_global_index(element_index, j_local)

                mij = self.mass[i, j]
                dij = self._artificial_diffusion[i, j]
                cij = self._discrete_gradient[i, j]

                ui = self._dof_vector[i]
                uj = self._dof_vector[j]
                fi = self._flux_approximation[i]
                fj = self._flux_approximation[j]
                dudt_i = self._low_cg_right_hand_side[i]
                dudt_j = self._low_cg_right_hand_side[j]
                umax_i = self.local_maximum[i]
                umax_j = self.local_maximum[j]
                umin_i = self.local_minimum[i]
                umin_j = self.local_minimum[j]

                fij = mij * (dudt_i - dudt_j) + dij * (ui - uj)
                wij = dij * (uj + ui) + (fi - fj) * cij

                fmin_ij = max(2 * dij * umin_i - wij, wij - 2 * dij * umax_j)
                fmax_ij = min(2 * dij * umax_i - wij, wij - 2 * dij * umin_j)
                fstar_ij = max(fmin_ij, min(fmax_ij, fij))

                corrected_flux[i] += fstar_ij
                corrected_flux[j] -= fstar_ij

        self[:] = (
            self._low_cg_right_hand_side.values
            + corrected_flux / self._lumped_mass.values
        )
