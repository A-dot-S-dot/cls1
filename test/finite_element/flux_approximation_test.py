from unittest import TestCase

import numpy as np

from finite_element.flux_approximation import *


class TestFluxApproximation(TestCase):
    test_dofs = [np.array([1, 0, 0, 0]), np.array([1, 2, 3, 4])]

    def test_approximation_linear_flux(self):
        flux = lambda x: x
        self._test_approximation(flux)

    def _test_approximation(self, flux):
        approximation = FluxApproximation(flux)
        for dofs in self.test_dofs:
            self.assertListEqual(list(approximation(dofs)), list(flux(dofs)))

    def test_approximation_burgers_flux(self):
        flux = lambda x: 1 / 2 * x**2
        self._test_approximation(flux)
