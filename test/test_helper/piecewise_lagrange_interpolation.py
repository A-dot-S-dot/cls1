from typing import Dict, Sequence

from mesh.interval import Interval
from numpy import nan, poly1d
from scipy.interpolate import lagrange


class PiecewiseLagrangeInterpolation:
    _polynomial: Dict[Interval, poly1d]

    def __init__(self):
        self._polynomial = {}

    def add_piecewise_polynomial(
        self,
        interpolation_points: Sequence[float],
        interpolation_values: Sequence[float],
        support: Interval,
    ):
        self._polynomial[support] = lagrange(interpolation_points, interpolation_values)

    def __call__(self, x: float) -> float:
        value = 0
        for support, interpolation in self._polynomial.items():
            if x in support:
                value = interpolation(x)
                break

        return value

    def derivative(self, x: float) -> float:
        value = 0

        for support, interpolation in self._polynomial.items():
            if x in support:
                if support.is_in_boundary(x):
                    value = nan
                else:
                    value = interpolation.deriv()(x)
                break

        return value
