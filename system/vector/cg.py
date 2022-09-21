from system.matrix.system_matrix import SystemMatrix

from .dof_vector import DOFVector
from .system_vector import SystemVector


class CGRightHandSide(SystemVector):
    """Right hand side of continuous Galerkin method r. To be more
    precise it is defined as following:

       Mr = A

    where M denotes mass marix and A the discrete flux gradient.

    """

    mass: SystemMatrix
    flux_gradient: SystemVector

    def __init__(self, dof_vector: DOFVector):
        SystemVector.__init__(self, dof_vector.element_space)
        dof_vector.register_observer(self)

    def assemble(self):
        self[:] = self.mass.inverse(self.flux_gradient.values)
