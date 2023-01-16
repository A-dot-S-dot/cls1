"""This module provides Transformation between diffrent simplices."""
from .interval import Interval


class AffineTransformation:
    """Mapping from reference_cell [0,1] to an arbitrary simplex."""

    def __call__(self, reference_cell_point: float, cell: Interval) -> float:
        return cell.length * reference_cell_point + cell.a

    def inverse(self, cell_point: float, cell: Interval) -> float:
        return (cell_point - cell.a) / cell.length

    def derivative(self, cell: Interval) -> float:
        return cell.length

    def inverse_derivative(self, cell: Interval) -> float:
        return 1 / cell.length
