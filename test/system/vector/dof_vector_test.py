from unittest import TestCase

import numpy as np
from system.vector import SystemVector
from system.vector.dof_vector import DOFVector

from ...test_helper import LINEAR_DOF_VECTOR, QUADRATIC_DOF_VECTOR


class ObserverVector(SystemVector):
    _dof_vector: DOFVector

    def set_observable(self, dof_vector: DOFVector):
        self._dof_vector = dof_vector
        dof_vector.register_observer(self)

    def update(self):
        self[:] = self._dof_vector[:]


class TestLinearDOFVector(TestCase):
    dof_vector = LINEAR_DOF_VECTOR
    dofs = [
        np.array((1, 0, 0, 0)),
        np.array((0, 1, 0, 0)),
        np.array((0, 0, 1, 0)),
        np.array((0, 0, 0, 1)),
    ]

    def test_set_dofs(self):
        for dofs in self.dofs:
            self.dof_vector.dofs = dofs
            self.assertListEqual(list(self.dof_vector), list(dofs))

    def test_assemble_observer(self):
        observer = ObserverVector(self.dof_vector.element_space)
        observer.set_observable(self.dof_vector)
        for dofs in self.dofs:
            self.dof_vector.dofs = dofs
            self.assertListEqual(list(self.dof_vector), list(observer))


class TestQuadraticDOFVector(TestLinearDOFVector):
    dof_vector = QUADRATIC_DOF_VECTOR
