from unittest import TestCase

from mesh import Interval
from mesh.uniform import UniformMesh

from numpy import sqrt
from quadrature.norm import L1Norm, L2Norm, LInfinityNorm


class TestL2Norm(TestCase):
    domain = Interval(-1, 1)
    mesh = UniformMesh(domain, 2)
    norm = L2Norm(mesh, 2)
    expected_norm = sqrt(2 / 3)

    def function(self, x: float) -> float:
        return x

    def test_norm(self):
        integral = self.norm(self.function)
        self.assertAlmostEqual(integral, self.expected_norm)


class TestL1Norm(TestL2Norm):
    domain = Interval(-1, 1)
    mesh = UniformMesh(domain, 2)
    norm = L1Norm(mesh, 1)
    expected_norm = 1


class TestLInfinityNorm(TestL2Norm):
    domain = Interval(-1, 1)
    mesh = UniformMesh(domain, 2)
    norm = LInfinityNorm(mesh, 10)
    expected_norm = 1
