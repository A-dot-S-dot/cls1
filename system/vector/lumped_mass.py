from fem import FiniteElementSpace
from system.matrix.mass import MassMatrix

from .system_vector import SystemVector


class LumpedMassVector(SystemVector):
    """Lumped mass system vector. It's entries are raw sums of mass matrix or
    Integral(bi), where bi is finite element basis.

    """

    _mass: MassMatrix

    def __init__(self, element_space: FiniteElementSpace):
        SystemVector.__init__(self, element_space)
        self._mass = MassMatrix(element_space)
        self.assemble()

    def assemble(self):
        self[:] = self._mass.values.sum(axis=1).reshape(self.dimension)
