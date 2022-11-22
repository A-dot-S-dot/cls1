import numpy as np
from pde_solver.discretization.finite_element import LagrangeFiniteElementSpace
from pde_solver.system_matrix.mass import MassMatrix

from .system_vector import SystemVector


class LumpedMassVector(SystemVector):
    """Lumped mass system vector. It's entries are raw sums of mass matrix or
    Integral(bi), where bi is finite element basis.

    """

    _lumped_mass: np.ndarray

    def __init__(self, element_space: LagrangeFiniteElementSpace):
        mass = MassMatrix(element_space)
        lumped_mass = mass().sum(axis=1)
        self._lumped_mass = lumped_mass.A1

    def __call__(self) -> np.ndarray:
        return self._lumped_mass
