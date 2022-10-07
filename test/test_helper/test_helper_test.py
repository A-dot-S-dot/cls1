from typing import Callable
from unittest import TestCase

import numpy as np
from numpy import cos, exp, sin
from pde_solver.mesh import Interval

from .discrete_derivative import discrete_derivative
from .piecewise_lagrange_interpolation import PiecewiseLagrangeInterpolation


class TestDiscreteDerivative(TestCase):
    def test_derivative(self):
        functions = [lambda x: x, lambda x: x**2, lambda x: sin(x), lambda x: exp(x)]
        derivatives = [lambda x: 1, lambda x: 2 * x, lambda x: cos(x), lambda x: exp(x)]
        for f, df in zip(functions, derivatives):
            for x in np.linspace(0, 1):
                self._test_derivative(f, df, x)

    def _test_derivative(
        self,
        function: Callable[[float], float],
        derivative: Callable[[float], float],
        x: float,
    ):
        self.assertAlmostEqual(discrete_derivative(function, x), derivative(x))


class TestPiecewiseLagrangeInterpolation(TestCase):
    supports = [Interval(0, 0.5), Interval(0.5, 1)]
    interpolation_points = [(0, 0.5), (0.5, 1)]
    interpolation_values = [(0, 1), (1, 0)]
    interpolation = PiecewiseLagrangeInterpolation()
    test_points = np.linspace(-1, 2)

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
        for points, values, support in zip(
            self.interpolation_points, self.interpolation_values, self.supports
        ):
            self.interpolation.add_piecewise_polynomial(points, values, support)

    def exact_interpolation(self, x: float) -> float:
        if x >= 0 and x <= 0.5:
            return 2 * x
        elif x >= 0.5 and x <= 1:
            return -2 * x + 2
        else:
            return 0

    def exact_interpolation_derivative(self, x: float) -> float:
        if x > 0 and x < 0.5:
            return 2
        elif x > 0.5 and x < 1:
            return -2
        elif x in {0, 0.5, 1}:
            return np.nan
        else:
            return 0

    def test_interpolation_values(self):
        for point in self.test_points:
            self.assertAlmostEqual(
                self.interpolation(point), self.exact_interpolation(point)
            )

    def test_interpolation_derivatives(self):
        for point in self.test_points:
            self.assertAlmostEqual(
                self.interpolation.derivative(point),
                self.exact_interpolation_derivative(point),
            )
