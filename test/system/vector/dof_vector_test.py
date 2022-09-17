from unittest import TestCase

from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh import Interval
from mesh.uniform import UniformMesh
from system.vector import DOFVector, SystemVector

from ...test_helper import LINEAR_DOF_VECTOR


class ObserverVector(SystemVector):
    _dof_vector: DOFVector

    def set_observable(self, dof_vector: DOFVector):
        self._dof_vector = dof_vector
        dof_vector.register_observer(self)

    def assemble(self):
        self[:] = self._dof_vector[:]


class TestLinearDOFVector(TestCase):
    dof_vector = LINEAR_DOF_VECTOR
    dofs = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]

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


class TestQuadraticDOFVector(TestCase):
    domain = Interval(0, 1)
    mesh = UniformMesh(domain, 2)
    element_space = LagrangeFiniteElementSpace(mesh, 2)
    vector = DOFVector(element_space)
    basis_coefficients = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
