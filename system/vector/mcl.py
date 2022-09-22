from scipy.sparse import spmatrix
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
        wij = dij*(uj+ui)-(fj-fi)*cij,
        fij = mij*(DuL_i-DuL_j)+dij*(ui-uj) (target flux)

    where mij denotes the mass matrix and DuL_i the right hand side of low order
    cg.

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

    def assemble(self):
        self[:] = 0

        for i in range(self.dimension):
            for j in self.element_space.get_neighbour_indices(i) - {i}:
                dij = self._artificial_diffusion[i, j]
                ui = self._dof_vector[i]
                uj = self._dof_vector[j]
                fi = self._flux_approximation[i]
                fj = self._flux_approximation[j]
                fij = self.mass[i, j] * (
                    self._low_cg_right_hand_side[i] - self._low_cg_right_hand_side[j]
                ) + dij * (ui - uj)
                cij = self._discrete_gradient[i, j]
                cji = self._discrete_gradient[j, i]
                wij = dij * (uj + ui) - cij * (fj - fi)
                wji = dij * (uj + ui) - cji * (fi - fj)
                fmax_ij = min(
                    2 * dij * self.local_maximum[i] - wij,
                    wji - 2 * dij * self.local_minimum[j],
                )
                fmin_ij = max(
                    2 * dij * self.local_minimum[i] - wij,
                    wji - 2 * dij * self.local_maximum[j],
                )
                fstar_ij = max(fmin_ij, min(fmax_ij, fij))

                self[i] += dij * (uj - ui) - (fj - fi) * cij + fstar_ij

        self[:] = self.values / self._lumped_mass.values
