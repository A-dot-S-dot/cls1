from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, Sequence, TypeVar

import defaults
import matplotlib.pyplot as plt
import numpy as np
from core.benchmark import Benchmark, NoExactSolutionError
from core.solver import Solver
from shallow_water.benchmark import ShallowWaterBenchmark
from tqdm.auto import tqdm

from .command import Command

T = TypeVar("T", float, np.ndarray)


class NothingToPlotError(Exception):
    ...


class Plotter(ABC, Generic[T]):
    _benchmark: Benchmark
    _figure: plt.Figure
    _grid: np.ndarray
    _save: Optional[str]
    _show: bool
    _plot_available = False

    def __init__(self, benchmark: Benchmark, mesh_size=None, save=None, show=True):
        self._benchmark = benchmark
        self._grid = np.linspace(
            self._benchmark.domain.a,
            self._benchmark.domain.b,
            mesh_size or defaults.PLOT_MESH_SIZE,
        )
        self._save = save
        self._show = show

    def set_suptitle(self, suptitle: str):
        self._figure.suptitle(suptitle, fontsize=14, fontweight="bold")

    @abstractmethod
    def add_initial_data(self):
        ...

    @abstractmethod
    def add_exact_solution(self):
        ...

    @abstractmethod
    def add_function(self, function: Callable[[float], float], *label: str):
        ...

    @abstractmethod
    def add_function_values(
        self, grid: np.ndarray, function_values: np.ndarray, *label: str
    ):
        ...

    def show(self):
        if self._plot_available:
            self._setup()

            if self._save:
                tqdm.write(f"Plot is saved in '{self._save}'.")
                self._figure.savefig(self._save)

            plt.show()
        else:
            raise NothingToPlotError

    @abstractmethod
    def _setup(self):
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class ScalarPlotter(Plotter[float]):
    _benchmark: Benchmark
    _axes: plt.Axes

    def __init__(self, benchmark: Benchmark, **kwargs):
        super().__init__(benchmark, **kwargs)
        self._figure, self._axes = plt.subplots()

    def add_initial_data(self):
        self.add_function(
            self._benchmark.initial_data,
            "$u_0$",
        )

    def add_exact_solution(self):
        self.add_function(
            self._benchmark.exact_solution_at_end_time,
            f"$u(t={self._benchmark.end_time:.1f})$",
        )

    def add_function(self, function: Callable[[float], float], label: str):
        function_values = np.array([function(x) for x in self._grid])
        self.add_function_values(self._grid, function_values, label)

    def add_function_values(
        self, grid: np.ndarray, function_values: np.ndarray, label: str
    ):
        self._axes.plot(grid, function_values, label=label)
        self._plot_available = True

    def _setup(self):
        self._axes.set_xlabel("x")
        self._axes.legend()


class ShallowWaterPlotter(Plotter[np.ndarray]):
    _benchmark: ShallowWaterBenchmark
    _figure: plt.Figure
    _height_axes: plt.Axes
    _discharge_axes: plt.Axes

    def __init__(self, benchmark: ShallowWaterBenchmark, **kwargs):
        Plotter.__init__(self, benchmark, **kwargs)
        self._figure, (self._height_axes, self._discharge_axes) = plt.subplots(1, 2)

    def set_suptitle(self, suptitle: str):
        self._figure.suptitle(suptitle, fontsize=14, fontweight="bold")

    def add_initial_data(self):
        self.add_function(self._benchmark.initial_data, "$h_0$", "$q_0$")

    def add_exact_solution(self):
        end_time = self._benchmark.end_time
        self.add_function(
            self._benchmark.exact_solution_at_end_time,
            f"$h+b(t={end_time:.1f})$",
            f"$q(t={end_time:.1f})$",
        )

    def add_function(self, function: Callable[[float], np.ndarray], *label: str):
        swe_values = np.array([function(x) for x in self._grid])
        self.add_function_values(self._grid, swe_values, *label)

    def add_function_values(
        self, grid: np.ndarray, swe_values: np.ndarray, *label: str
    ):
        height = swe_values.T[0]
        discharge = swe_values.T[1]

        height_label = label[0]

        if len(label) == 1:
            discharge_label = label[0]
        else:
            discharge_label = label[1]

        self.add_height(grid, height, height_label)
        self.add_discharge(grid, discharge, discharge_label)

        self._plot_available = True

    def add_height(self, grid: np.ndarray, height: np.ndarray, label: str):
        total_height = height + [self._benchmark.topography(x) for x in grid]
        self._height_axes.plot(grid, total_height, label=label)

    def add_discharge(self, grid: np.ndarray, discharge: np.ndarray, label: str):
        self._discharge_axes.plot(grid, discharge, label=label)

    def _setup(self):
        self._add_topography()

        self._height_axes.set_xlabel("x")
        self._height_axes.set_ylabel("h+b")
        self._height_axes.legend()

        self._discharge_axes.set_xlabel("x")
        self._discharge_axes.set_ylabel("discharge")
        self._discharge_axes.legend()

    def _add_topography(self):
        topography_values = np.array(
            [self._benchmark.topography(x) for x in self._grid]
        )
        self._height_axes.plot(self._grid, topography_values, label="$b$")


class Plot(Command):
    _benchmark: Benchmark
    _solver: Sequence[Solver]
    _plotter: Plotter
    _initial: bool

    def __init__(
        self,
        benchmark: Benchmark,
        solver: Solver | Sequence[Solver],
        plotter: Plotter,
        initial=False,
    ):
        self._benchmark = benchmark
        self._plotter = plotter
        self._initial = initial

        if isinstance(solver, Solver):
            self._solver = [solver]
        else:
            self._solver = solver

    def execute(self):
        self._add_plots()
        self._plotter.set_suptitle(f"T={self._benchmark.end_time}")

        try:
            self._plotter.show()
        except NothingToPlotError:
            tqdm.write("WARNING: Nothing to plot...")

    def _add_plots(self):
        self._add_exact_solution()
        self._add_discrete_solutions()

        if self._initial:
            self._plotter.add_initial_data()

    def _add_exact_solution(self):
        try:
            self._plotter.add_exact_solution()
        except NoExactSolutionError as error:
            tqdm.write("WARNING: " + str(error))

    def _add_discrete_solutions(self):
        for solver in self._solver:
            solution = solver._solution
            self._plotter.add_function_values(
                solution.grid, solution.value, solver.short
            )
