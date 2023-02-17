from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import numpy as np
from .mesh import Interval


class Quadrature(ABC):
    _domain: Interval
    _nodes: List[float]
    _weights: np.ndarray

    @property
    def domain(self):
        return self._domain

    @property
    def nodes(self) -> List[float]:
        return self._nodes

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def degree(self) -> int:
        return len(self._nodes)

    @abstractmethod
    def integrate(self, f: Callable[[float], float]):
        ...


class SpecificGaussianQuadrature(Quadrature):
    """Gaussian quadrature on the interval [-1,1]."""

    _domain = Interval(-1, 1)
    _nodes: List[float]
    _quadrature_degree: int

    def __init__(self, quadrature_degree: int) -> None:
        assert quadrature_degree > 0, f"{quadrature_degree} is not a positive integer."
        self._quadrature_degree = quadrature_degree
        self._nodes_weights = self._calculate_nodes_weights()

    def _calculate_nodes_weights(self) -> List[Tuple[float, float]]:
        beta = np.zeros(self._quadrature_degree)
        t = np.zeros((self._quadrature_degree, self._quadrature_degree))

        for i in range(self._quadrature_degree - 1):
            beta[i] = 1 / 2 * (1 - (2 * (i + 1)) ** (-2)) ** (-0.5)
            t[i, i + 1] = beta[i]
            t[i + 1, i] = beta[i]

        nodes, v = np.linalg.eig(t)
        weights = []
        for i in range(self._quadrature_degree):
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

    def integrate(self, function: Callable[[float], float]) -> float:
        quadrature_nodes_values = np.array([function(node) for node in self.nodes])

        return np.dot(self.weights, quadrature_nodes_values)


class GaussianQuadrature(SpecificGaussianQuadrature):
    """Gaussian quadrature for an arbitrary given interval."""

    _factor: float
    _shift: float

    def __init__(self, quadrature_degree: int, interval: Interval):
        SpecificGaussianQuadrature.__init__(self, quadrature_degree)

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


class LocalElementQuadrature(GaussianQuadrature):
    """Gaussian Quadrature on the standard element [0,1]."""

    def __init__(self, quadrature_degree: int):
        """The quadrature is 2*NODES_NUMBER-1 exact."""
        GaussianQuadrature.__init__(self, quadrature_degree, Interval(0, 1))
