from typing import List, Tuple

import numpy as np
from math_type import ScalarFunction
from mesh import Interval

from .abstracts import Quadrature


class GaussianQuadrature(Quadrature):
    """Gaussian quadrature on the interval [-1,1]."""

    _domain = Interval(-1, 1)
    _nodes: List[float]
    _nodes_number: int

    def __init__(self, nodes_number: int) -> None:
        assert nodes_number > 0, f"{nodes_number} is not a positive integer."
        self._nodes_number = nodes_number
        self._nodes_weights = self._calculate_nodes_weights()

    def _calculate_nodes_weights(self) -> List[Tuple[float, float]]:
        beta = np.zeros(self._nodes_number)
        t = np.zeros((self._nodes_number, self._nodes_number))

        for i in range(self._nodes_number - 1):
            beta[i] = 1 / 2 * (1 - (2 * (i + 1)) ** (-2)) ** (-0.5)
            t[i, i + 1] = beta[i]
            t[i + 1, i] = beta[i]

        nodes, v = np.linalg.eig(t)
        weights = []
        for i in range(self._nodes_number):
            weights.append(2 * (v[0, i] ** 2))

        nodes_weights = list(zip(nodes, weights))
        nodes_weights.sort()

        self._build_nodes_weights(nodes_weights)

        return nodes_weights

    def _build_nodes_weights(self, nodes_weights: List[Tuple[float, float]]):
        self._nodes = [node for node, _ in nodes_weights]
        self._weights = np.array([weight for _, weight in nodes_weights])

    @property
    def nodes(self) -> List[float]:
        return self._nodes

    def integrate(self, function: ScalarFunction) -> float:
        quadrature_nodes_values = np.array([function(node) for node in self.nodes])

        return np.dot(self.weights, quadrature_nodes_values)


class GaussianQuadratureGeneralized(GaussianQuadrature):
    """Gaussian quadrature for an arbitrary given interval."""

    _factor: float
    _shift: float

    def __init__(self, nodes_number: int, interval: Interval):
        GaussianQuadrature.__init__(self, nodes_number)

        if self.domain != interval:
            self._domain = interval

            self._factor = (self.domain.b - self.domain.a) / 2
            self._shift = (self.domain.a + self.domain.b) / 2

            self._transform_nodes()
            self._transform_weights()

    def _transform_nodes(self):
        transformed_nodes = []

        for node in self.nodes:
            transformed_node = self._factor * node + self._shift
            transformed_nodes.append(transformed_node)

        self._nodes = transformed_nodes

    def _transform_weights(self):
        transformed_weights = []

        for weight in self.weights:
            transformed_weight = self._factor * weight
            transformed_weights.append(transformed_weight)

        self._weights = np.array(transformed_weights)
