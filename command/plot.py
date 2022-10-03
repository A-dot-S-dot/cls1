import time
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Callable, Generic, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from benchmark import Benchmark
from factory.pde_solver_factory import PDESolverFactory
from mesh import Interval
from pde_solver.solver_components import SolverComponents
from tqdm import tqdm

from .command import Command

T = TypeVar("T", float, np.ndarray)


class FunctionPlotter(ABC, Generic[T]):
    _grid: np.ndarray
    _suptitle: str

    def set_grid(self, interval: Interval, dof_number: int = 0):

        grid_number = max(500, dof_number * 2)
        self._grid = np.linspace(interval.a, interval.b, grid_number)

    @property
    def title(self) -> str:
        ...

    @title.setter
    def title(self, title: str):
        plt.title(title)

    @property
    def suptitle(self) -> str:
        ...

    @suptitle.setter
    def suptitle(self, suptitle: str):
        plt.suptitle(suptitle, fontsize=14, fontweight="bold")

    @abstractmethod
    def add_function(self, function: Callable[[float], T], function_label: str):
        ...

    def save(self, path="output/plot.png"):
        self._setup()
        plt.savefig(path)

    def show(self):
        self._setup()
        plt.show()
        plt.close()

    def _setup(self):
        plt.xlabel("x")
        plt.legend()


class ScalarFunctionPlotter(FunctionPlotter[float]):
    def add_function(self, function: Callable[[float], float], function_label: str):
        function_values = np.array([function(x) for x in self._grid])
        plt.plot(self._grid, function_values, label=function_label)


class PlotCommand(Command):
    _args: Namespace
    _components: SolverComponents
    _benchmark: Benchmark
    _plotter: FunctionPlotter

    def __init__(self, args: Namespace):
        self._args = args
        self._components = SolverComponents(args)
        self._benchmark = self._components.benchmark
        self._build_plotter()

    def _build_plotter(self):
        self._plotter = ScalarFunctionPlotter()

        domain = self._benchmark.domain
        dofs_number = self._get_maximal_polynomial_degree() * self._components.mesh_size

        self._plotter.set_grid(domain, dofs_number)

    def _get_maximal_polynomial_degree(self) -> int:
        maximal_polynomial_degree = 0
        if self._args.solver:
            for solver_args in self._args.solver:
                maximal_polynomial_degree = max(
                    solver_args.polynomial_degree, maximal_polynomial_degree
                )

        return maximal_polynomial_degree

    def execute(self):
        self._add_functions()
        self._plotter.title = f"{len(self._components.mesh)} elements"

        if not self._args.quite:
            self._plotter.show()

    def _add_functions(self):
        self._add_exact_solution()
        self._add_discrete_solutions()

    def _add_exact_solution(self):
        benchmark = self._components.benchmark

        if benchmark.has_exact_solution():
            self._plotter.add_function(
                benchmark.exact_solution_at_end_time,
                f"$u(\cdot, {self._benchmark.end_time:.1f})$",
            )
        elif len(self._components.solver_factories) == 0:
            print("WARNING: Nothing to do...")

    def _add_discrete_solutions(self):
        for solver_factory in tqdm(
            self._components.solver_factories,
            desc="Calculate solutions",
            unit="solver",
            position=0,
            leave=False,
        ):
            self._add_discrete_solution(solver_factory)

    def _add_discrete_solution(self, solver_factory: PDESolverFactory):
        solver = solver_factory.solver

        start_time = time.time()
        solver.solve()

        label = solver_factory.plot_label
        tqdm.write(
            f"Solved {solver_factory.info} with {solver_factory.dofs} DOFs and {solver.time_steps} time steps in {time.time()-start_time:.2f}s."
        )

        self._plotter.add_function(solver_factory.discrete_solution, label)
