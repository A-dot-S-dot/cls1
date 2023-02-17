from abc import ABC, abstractmethod
from typing import Iterator, Sequence

import numpy as np


class Interval:
    a: float
    b: float

    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)

        assert self.a < self.b, f"{a} is not strictly less than {b}"

    def __str__(self) -> str:
        return f"[{self.a}, {self.b}]"

    @property
    def length(self) -> float:
        return self.b - self.a

    @property
    def center(self) -> float:
        return (self.b + self.a) / 2

    def __hash__(self) -> int:
        return hash((self.a, self.b))

    def __eq__(self, other) -> bool:
        return self.a == other.a and self.b == other.b

    def __contains__(self, x: float) -> bool:
        return x >= self.a and x <= self.b

    def is_in_inner(self, x: float) -> bool:
        return x > self.a and x < self.b

    def is_in_boundary(self, x: float) -> bool:
        return x == self.a or x == self.b


class Mesh(ABC):
    """Mesh class for coherent meshes."""

    _domain: Interval
    _cells: Sequence[Interval]

    @property
    def domain(self) -> Interval:
        return self._domain

    @property
    def step_length(self) -> float:
        simplices_step_length = [simplex.length for simplex in self._cells]
        return max(*simplices_step_length)

    @property
    @abstractmethod
    def space_steps(sefl) -> np.ndarray:
        ...

    def __iter__(self) -> Iterator[Interval]:
        return iter(self._cells)

    def __len__(self) -> int:
        return len(self._cells)

    def __getitem__(self, index: int) -> Interval:
        return self._cells[index]

    def __eq__(self, other) -> bool:
        return self._cells == other._cells

    def index(self, cell: Interval) -> int:
        return self._cells.index(cell)

    def find_cell_indices(self, point: float) -> Sequence[int]:
        indices = []
        for cell in self._cells:
            if point in cell:
                indices.append(self.index(cell))

        return indices

    def find_cells(self, point: float) -> Sequence[Interval]:
        return [self[index] for index in self.find_cell_indices(point)]

    @abstractmethod
    def refine(self, refine_degree: int = 2) -> "Mesh":
        ...

    @abstractmethod
    def coarsen(self, coarsening_degree: int) -> "Mesh":
        ...


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

    def refine(self, refine_degree: int = 2) -> "UniformMesh":
        return UniformMesh(self.domain, refine_degree * len(self))

    def coarsen(self, coarsening_degree: int) -> "UniformMesh":
        assert (
            len(self) % coarsening_degree == 0
        ), f"Mesh of size {len(self)} cannot be coarsened with degree of {coarsening_degree}"

        return UniformMesh(self.domain, len(self) // coarsening_degree)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + f"(domain={self.domain}, mesh_size={len(self)})"
        )


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
