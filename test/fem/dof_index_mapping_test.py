from unittest import TestCase

from mesh import Interval
from mesh.uniform import UniformMesh

from fem.dof_index_mapping import PeriodicDOFIndexMapping
from fem.lagrange.local_lagrange import LocalLagrangeBasis


class TestPeriodicDOFIndexMappingLinearLocalBasis(TestCase):
    element_number = 3
    interval = Interval(0, 1)
    mesh = UniformMesh(interval, element_number)
    local_bases = [LocalLagrangeBasis(p + 1) for p in range(7)]
    inputs = [
        (0, 0),
        (2, 1),
        (1, 0),
    ]
    expected_outputs = [0, 0, 1]

    def test_inner_point(self):
        for local_basis in self.local_bases:
            index_mapping = PeriodicDOFIndexMapping(self.mesh, len(local_basis))
            input = (1, 1)
            expected_output = local_basis.polynomial_degree + 1
            self.assertEqual(index_mapping(*input), expected_output)

    def test_boundary_condition(self):
        for local_basis in self.local_bases:
            index_mapping = PeriodicDOFIndexMapping(self.mesh, len(local_basis))
            for input in [(0, 0), (2, local_basis.polynomial_degree)]:
                self.assertEqual(index_mapping(*input), 0)
