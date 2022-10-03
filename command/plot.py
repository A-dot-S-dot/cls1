import time
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Callable, Generic, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from benchmark import Benchmark
from benchmark.abstract import NoExactSolutionError
from factory.pde_solver_factory import PDESolverFactory
from mesh import Interval
from pde_solver.solver_components import SolverComponents
from tqdm import tqdm

from .command import Command

T = TypeVar("T", float, np.ndarray)


class NothingToPlotError(Exception):
    ...


class SolutionPlotter(ABC, Generic[T]):
    _figure = plt.figure()
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
    def add_initial_data(self):
        ...

    @abstractmethod
    def add_exact_solution(self):
        ...

    @abstractmethod
    def add_function(self, function: Callable[[float], T], label: str):
        ...

    def show(self):
        if self._figure.get_axes():
            self._setup()
            plt.show()
            plt.close()

        else:
            raise NothingToPlotError

    @abstractmethod
    def _setup(self):
        ...


class ScalarFunctionPlotter(SolutionPlotter[float]):
    _axes: plt.Axes
    _benchmark: Benchmark

    def __init__(self, benchmark: Benchmark):
        self._axes = self._figure.subplots()
        self._benchmark = benchmark

    def add_initial_data(self):
        self.add_function(
            self._benchmark.initial_data,
            "$u_0$",
        )

    def add_exact_solution(self):
        self.add_function(
            self._benchmark.exact_solution_at_end_time,
            f"$u(\cdot, {self._benchmark.end_time:.1f})$",
        )

    def add_function(self, function: Callable[[float], float], label: str):
        function_values = np.array([function(x) for x in self._grid])
        self._axes.plot(self._grid, function_values, label=label)

    def _setup(self):
        self._axes.set_xlabel("x")
        self._axes.legend()


class PlotCommand(Command):
    _args: Namespace
    _components: SolverComponents
    _plotter: SolutionPlotter

    def __init__(self, args: Namespace):
        self._args = args
        self._components = SolverComponents(args)
        self._build_plotter()

    def _build_plotter(self):
        benchmark = self._components.benchmark
        domain = benchmark.domain
        dofs_number = self._get_cell_resolution() * self._components.mesh_size

        self._plotter = ScalarFunctionPlotter(benchmark)
        self._plotter.set_grid(domain, dofs_number)

    def _get_cell_resolution(self) -> int:
        cell_resolution = 1
        for solver_factory in self._components.solver_factories:
            cell_resolution = max(
                cell_resolution, solver_factory.cell_quadrature_degree
            )

        return cell_resolution

    def execute(self):
        self._add_plots()
        self._plotter.title = f"{len(self._components.mesh)} cells"

        if not self._args.plot.quite:
            try:
                self._plotter.show()
            except NothingToPlotError:
                tqdm.write("WARNING: Nothing to plot...")

    def _add_plots(self):
        self._add_exact_solution()
        self._add_discrete_solutions()

        if self._args.plot.initial:
            self._plotter.add_initial_data()

    def _add_exact_solution(self):
        try:
            self._plotter.add_exact_solution()
        except NoExactSolutionError as error:
            tqdm.write("WARNING: " + str(error))

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
