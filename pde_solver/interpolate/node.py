from typing import Sequence

import numpy as np
from custom_type import ScalarFunction

from .interpolator import Interpolator


class NodeValuesInterpolator(Interpolator):
    """Interpolate functions by calculating values on given nodes."""

    _nodes: Sequence[float]

    def __init__(self, *nodes: float):
        self._nodes = nodes

    def interpolate(self, f: ScalarFunction) -> np.ndarray:
        return np.array([f(node) for node in self._nodes])
