"""This module provides different types for typing."""
from typing import Callable

from numpy.typing import ArrayLike


ScalarFunction = Callable[[float], float]
MultidimensionalFunction = Callable[[float], ArrayLike]
