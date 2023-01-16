from unittest import TestCase

from base.quadrature.local import LocalElementQuadrature


class TestLocalElementQuadrature(TestCase):
    def test_local_quadrature(self):
        local_element_quadrature = LocalElementQuadrature(2)

        functions = [
            lambda _: 1,
            lambda x: x,
            lambda x: x**2,
            lambda x: x**3,
        ]

        integrals = [1, 1 / 2, 1 / 3, 1 / 4]

        for f, integral in zip(functions, integrals):
            self.assertAlmostEqual(local_element_quadrature.integrate(f), integral)
