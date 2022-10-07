from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Dict, Generic, Iterable, TypeVar

import numpy as np
from benchmark import Benchmark
from pde_solver.mesh import Mesh
from pde_solver.solver import PDESolver

T = TypeVar("T", np.ndarray, float)


class PDESolverFactory(ABC, Generic[T]):
    attributes: Namespace
    problem_name: str
    mesh: Mesh
    benchmark: Benchmark

    _solver: PDESolver

    @property
    def solver(self) -> PDESolver:
        self._setup_solver()

        return self._solver

    @abstractmethod
    def _setup_solver(self):
        ...

    @property
    @abstractmethod
    def grid(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def cell_quadrature_degree(self) -> int:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def plot_label(self) -> Iterable[str]:
        ...

    @property
    @abstractmethod
    def eoc_title(self) -> str:
        ...

    @property
    @abstractmethod
    def tqdm_kwargs(self) -> Dict:
        ...

    @property
    @abstractmethod
    def info(self) -> str:
        ...
