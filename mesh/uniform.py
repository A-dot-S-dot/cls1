from typing import Sequence
from .mesh import Mesh

import numpy as np

from .interval import Interval


class UniformMesh(Mesh):
    _step_length: float
    _domain: Interval
    _grid: np.ndarray
    _simplices: Sequence[Interval]

    def __init__(self, domain: Interval, mesh_size: int) -> None:
        if mesh_size <= 0:
            raise ValueError(f"{mesh_size} is not positive")

        self._domain = domain

        nodes_number = mesh_size + 1
        self._grid, step_length = np.linspace(
            domain.a, domain.b, nodes_number, retstep=True
        )

        self._step_length = float(step_length)

        self._build_simplices()

    def _build_simplices(self):
        self._simplices = []
        for index in range(len(self)):
            self._simplices.append(Interval(self._grid[index], self._grid[index + 1]))

    @property
    def step_length(self) -> float:
        return self._step_length

    def __eq__(self, other) -> bool:
        return self.step_length == other.step_length and self.domain == other.domain

    def refine(self):
        new_mesh_size = 2 * len(self)

        return UniformMesh(self.domain, new_mesh_size)
