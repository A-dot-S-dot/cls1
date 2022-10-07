from .mesh import Mesh

import numpy as np

from .interval import Interval


class UniformMesh(Mesh):
    _step_length: float
    _space_steps: np.ndarray

    def __init__(self, domain: Interval, mesh_size: int) -> None:
        self._domain = domain

        grid = self._build_grid(mesh_size)
        self._build_cells(grid)
        self._build_space_steps()

    def _build_grid(self, mesh_size) -> np.ndarray:
        nodes_number = mesh_size + 1
        grid, step_length = np.linspace(
            self.domain.a, self.domain.b, nodes_number, retstep=True
        )
        self._step_length = float(step_length)

        return grid

    def _build_cells(self, grid: np.ndarray):
        self._cells = []

        for index in range(len(grid) - 1):
            self._cells.append(Interval(grid[index], grid[index + 1]))

    def _build_space_steps(self):
        self._space_steps = np.array(len(self) * [self.step_length])

    @property
    def step_length(self) -> float:
        return self._step_length

    @property
    def space_steps(self) -> np.ndarray:
        return self._space_steps

    def __eq__(self, other) -> bool:
        return self.step_length == other.step_length and self.domain == other.domain

    def refine(self) -> "UniformMesh":
        new_mesh_size = 2 * len(self)

        return UniformMesh(self.domain, new_mesh_size)
