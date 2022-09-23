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
        fmin_ij = min(2*dij*umin_i-wij, wji-2*dji*umax_j),
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
        corrected_flux = self._build_corrected_flux()
        self[:] = self._low_cg_right_hand_side.values + corrected_flux

    def _build_corrected_flux(self) -> spmatrix:
        target_flux = self._build_target_flux()
        wij = self._build_wij()
        fmax = self._build_fmax(wij)
        fmin = self._build_fmin(wij)

        fstar = fmin.maximum(target_flux.minimum(fmax))
        fstar = fstar.tolil()
        fstar.setdiag(0)

        return fstar.tocsr().sum(axis=1).reshape(self.dimension)

    def _build_target_flux(self) -> spmatrix:
        return (
            self.mass.multiply_column(self._low_cg_right_hand_side)
            - self.mass.multiply_row(self._low_cg_right_hand_side)
            + self._artificial_diffusion.multiply_column(self._dof_vector)
            - self._artificial_diffusion.multiply_row(self._dof_vector)
        )

    def _build_wij(self) -> spmatrix:
        return (
            self._artificial_diffusion.multiply_column(self._dof_vector)
            + self._artificial_diffusion.multiply_row(self._dof_vector)
            - self._discrete_gradient.multiply_row(self._flux_approximation)
            + self._discrete_gradient.multiply_column(self._flux_approximation)
        )

    def _build_fmax(self, wij: spmatrix) -> spmatrix:
        value1 = (
            2 * self._artificial_diffusion.multiply_column(self.local_maximum) - wij
        )
        value2 = wij.transpose() - 2 * self._artificial_diffusion.multiply_row(
            self.local_minimum
        )

        return value1.minimum(value2)

    def _build_fmin(self, wij: spmatrix) -> spmatrix:
        value1 = (
            2 * self._artificial_diffusion.multiply_column(self.local_minimum) - wij
        )
        value2 = wij.transpose() - 2 * self._artificial_diffusion.multiply_row(
            self.local_maximum
        )

        return value1.maximum(value2)
