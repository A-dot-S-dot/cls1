import numpy as np
from core import LagrangeSpace, SystemVector

from .mass import MassMatrix


class LumpedMassVector(SystemVector):
    """Lumped mass system vector. It's entries are raw sums of mass matrix or
    Integral(bi), where bi is finite element basis.

    """

    _lumped_mass: np.ndarray

    def __init__(self, element_space: LagrangeSpace):
        mass = MassMatrix(element_space)
        lumped_mass = mass().sum(axis=1)
        self._lumped_mass = lumped_mass.A1

    def __call__(self) -> np.ndarray:
        return self._lumped_mass
