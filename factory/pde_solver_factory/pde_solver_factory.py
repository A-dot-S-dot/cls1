from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Dict, Generic, Iterable, TypeVar

import numpy as np
from benchmark import Benchmark
from pde_solver.mesh import Mesh
from pde_solver.solver import PDESolver
from pde_solver.solver_space import SolverSpace

T = TypeVar("T", np.ndarray, float)


class PDESolverFactory(ABC, Generic[T]):
    attributes: Namespace
    problem_name: str
    mesh: Mesh
    benchmark: Benchmark

    _solver: PDESolver
    _solver_space: SolverSpace
    _plot_label: Iterable[str]

    @property
    def solver(self) -> PDESolver:
        self._setup_solver()

        return self._solver

    @abstractmethod
    def _setup_solver(self):
        ...

    @property
    def solver_space(self) -> SolverSpace:
        return self._solver_space

    @property
    @abstractmethod
    def grid(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def cell_quadrature_degree(self) -> int:
        ...

    @property
    def dimension(self) -> int:
        return self._solver_space.dimension

    @property
    def plot_label(self) -> Iterable[str]:
        if self.attributes.label:
            return [self.attributes.label]
        else:
            return self._plot_label

    @property
    def eoc_title(self) -> str:
        title = self.info
        return title + "\n" + "-" * len(title)

    @property
    @abstractmethod
    def info(self) -> str:
        ...

    @property
    @abstractmethod
    def tqdm_kwargs(self) -> Dict:
        ...
