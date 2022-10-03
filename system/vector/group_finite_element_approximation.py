from typing import Callable

import numpy as np

from .dof_vector import DOFVector


class GroupFiniteElementApproximation(DOFVector):
    """Group Finite Element Approximation (GFE) of f(v) with finite elements,
    where f is a flux and v a finite element. To be more precise it is defined
    as following:

        F = sum(fi*bi)

    where fi = f(vi) and bi denotes the basis elements of the finite element
    space.

    Not assembled by default.

    """

    _dof_vector: DOFVector
    _flux: Callable[[np.ndarray], np.ndarray]

    def __init__(
        self,
        dof_vector: DOFVector,
        flux: Callable[[np.ndarray], np.ndarray],
    ):
        DOFVector.__init__(self, dof_vector.element_space)
        self._dof_vector = dof_vector
        self._flux = flux

        dof_vector.register_observer(self)

    def update(self):
        self.dofs = self._flux(self._dof_vector.values)
