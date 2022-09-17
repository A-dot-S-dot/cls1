"""This module provides Transformation between diffrent simplices."""
from .interval import Interval


class AffineTransformation:
    """Mapping from standard simplex [0,1] to an arbitrary simplex."""

    def __call__(self, standard_simplex_point: float, simplex: Interval) -> float:
        return simplex.length * standard_simplex_point + simplex.a

    def inverse(self, simplex_point: float, simplex: Interval) -> float:
        return (simplex_point - simplex.a) / simplex.length

    def derivative(self, simplex: Interval) -> float:
        return simplex.length

    def inverse_derivative(self, simplex: Interval) -> float:
        return 1 / simplex.length
