from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Optional, TypeVar

import core
import defaults
import matplotlib.pyplot as plt
import numpy as np
import finite_volume.shallow_water as swe
from tqdm.auto import tqdm

from .calculate import Calculate
from .command import Command

T = TypeVar("T", float, np.ndarray)


class NothingToPlotError(Exception):
    ...


class Plotter(ABC, Generic[T]):
    _benchmark: core.Benchmark
    _figure: plt.Figure
    _grid: np.ndarray
    _save: Optional[str]
    _show: bool
    _plot_available = False

    def __init__(self, benchmark: core.Benchmark, mesh_size=None, save=None, show=True):
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
                self._figure.savefig(self._save)
                tqdm.write(f"Plot is saved in '{self._save}'.")

            plt.show() if self._show else plt.close()
        else:
            raise NothingToPlotError

    @abstractmethod
    def _setup(self):
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class ScalarPlotter(Plotter[float]):
    _benchmark: core.Benchmark
    _axes: plt.Axes

    def __init__(self, benchmark: core.Benchmark, **kwargs):
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
    _benchmark: swe.ShallowWaterBenchmark
    _figure: plt.Figure
    _height_axes: plt.Axes
    _discharge_axes: plt.Axes
    _velocity_axes: plt.Axes

    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        Plotter.__init__(self, benchmark, **kwargs)
        self._figure, (
            self._height_axes,
            self._discharge_axes,
            self._velocity_axes,
        ) = plt.subplots(1, 3)

    def set_suptitle(self, suptitle: str):
        self._figure.suptitle(suptitle, fontsize=14, fontweight="bold")

    def add_initial_data(self):
        self.add_function(self._benchmark.initial_data, "$h_0$", "$q_0$", "$u_0$")

    def add_exact_solution(self):
        end_time = self._benchmark.end_time
        self.add_function(
            self._benchmark.exact_solution_at_end_time,
            f"$h+b(t={end_time:.1f})$",
            f"$q(t={end_time:.1f})$",
            f"$u(t={end_time:.1f})$",
        )

    def add_function(self, function: Callable[[float], np.ndarray], *label: str):
        swe_values = np.array([function(x) for x in self._grid])
        self.add_function_values(self._grid, swe_values, *label)

    def add_function_values(self, grid: np.ndarray, values: np.ndarray, *label: str):
        height_label = label[0]

        if len(label) == 1:
            discharge_label = label[0]
            velocity_label = label[0]
        else:
            discharge_label = label[1]
            velocity_label = label[2]

        self.add_height(grid, values, height_label)
        self.add_discharge(grid, values, discharge_label)
        self.add_velocity(grid, values, velocity_label)

        self._plot_available = True

    def add_height(self, grid: np.ndarray, values: np.ndarray, label: str):
        height = swe.get_height(values)
        total_height = height + [self._benchmark.bathymetry(x) for x in grid]
        self._height_axes.plot(grid, total_height, label=label)

    def add_discharge(self, grid: np.ndarray, values: np.ndarray, label: str):
        discharge = swe.get_discharge(values)
        self._discharge_axes.plot(grid, discharge, label=label)

    def add_velocity(self, grid: np.ndarray, values: np.ndarray, label: str):
        velocity = swe.get_velocity(values)
        self._velocity_axes.plot(grid, velocity, label=label)

    def _setup(self):
        self._add_bathymetry()

        self._height_axes.set_xlabel("x")
        self._height_axes.set_ylabel("h+b")
        self._height_axes.legend()

        self._discharge_axes.set_xlabel("x")
        self._discharge_axes.set_ylabel("discharge")
        self._discharge_axes.legend()

        self._velocity_axes.set_xlabel("x")
        self._velocity_axes.set_ylabel("velocity")
        self._velocity_axes.legend()

    def _add_bathymetry(self):
        bathymetry_values = np.array(
            [self._benchmark.bathymetry(x) for x in self._grid]
        )
        self._height_axes.plot(self._grid, bathymetry_values, label="$b$")


class Plot(Command):
    _benchmark: core.Benchmark
    _solver: List[core.Solver]
    _plotter: Plotter
    _initial: bool
    _solver_executed: bool
    _write_warnings: bool

    def __init__(
        self,
        benchmark: core.Benchmark,
        solver: core.Solver | List[core.Solver],
        plotter: Plotter,
        initial=False,
        solver_executed=False,
        write_warnings=True,
    ):
        self._benchmark = benchmark
        self._plotter = plotter
        self._initial = initial
        self._solver_executed = solver_executed
        self._write_warnings = write_warnings

        if isinstance(solver, core.Solver):
            self._solver = [solver]
        else:
            self._solver = solver

    def execute(self):
        if not self._solver_executed:
            self._calculate_solutions()

        self._add_plots()
        self._plotter.set_suptitle(f"T={self._benchmark.end_time}")

        try:
            self._plotter.show()
        except NothingToPlotError:
            if self._write_warnings:
                tqdm.write("WARNING: Nothing to plot...")

    def _calculate_solutions(self):
        tqdm.write("\nCalculate solutions")
        tqdm.write("-------------------")
        for solver in tqdm(self._solver, desc="Calculate", unit="solver", leave=False):
            try:
                Calculate(solver).execute()
            except Exception as error:
                if self._write_warnings:
                    tqdm.write(f"WARNING: {str(error)}")

    def _delete_not_solved_solutions(self):
        accepted_solver = []
        for solver in self._solver:
            if solver.solution.time == self._benchmark.end_time:
                accepted_solver.append(solver)

        self._solver = accepted_solver

    def _add_plots(self):
        self._add_exact_solution()
        self._add_discrete_solutions()

        if self._initial:
            self._plotter.add_initial_data()

    def _add_exact_solution(self):
        try:
            self._plotter.add_exact_solution()
        except Exception as error:
            if self._write_warnings:
                tqdm.write("WARNING: " + str(error))

    def _add_discrete_solutions(self):
        for solver in self._solver:
            solution = solver.solution
            self._plotter.add_function_values(
                solution.grid, solution.value, solver.short
            )
