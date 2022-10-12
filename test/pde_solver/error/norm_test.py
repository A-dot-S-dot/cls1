from unittest import TestCase

from numpy import sqrt
from pde_solver.mesh import Interval
from pde_solver.mesh.uniform import UniformMesh
from pde_solver.error.norm import L1Norm, L2Norm, LInfinityNorm


class TestL2Norm(TestCase):
    domain = Interval(-1, 1)
    mesh = UniformMesh(domain, 2)
    norm = L2Norm(2)
    norm.mesh = mesh
    expected_norm = sqrt(2 / 3)

    def function(self, x: float) -> float:
        return x

    def test_norm(self):
        integral = self.norm(self.function)
        self.assertAlmostEqual(integral, self.expected_norm)


class TestL1Norm(TestL2Norm):
    domain = Interval(-1, 1)
    mesh = UniformMesh(domain, 2)
    norm = L1Norm(1)
    norm.mesh = mesh
    expected_norm = 1


class TestLInfinityNorm(TestL2Norm):
    domain = Interval(-1, 1)
    mesh = UniformMesh(domain, 2)
    norm = LInfinityNorm(10)
    norm.mesh = mesh
    expected_norm = 1
