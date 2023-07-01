import argparse
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, List, Optional, TypeVar

import core
import defaults
import finite_volume.shallow_water as swe
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .calculate import Calculate, CalculateParser
from .command import Command

T = TypeVar("T", float, np.ndarray)


class NothingToPlotError(Exception):
    ...


class Plotter(ABC, Generic[T]):
    _benchmark: core.Benchmark
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

    @abstractmethod
    def show(self):
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

    def show(self):
        if self._plot_available:
            self._axes.set_xlabel("x")
            self._axes.legend()

            if self._save:
                self._figure.savefig(self._save)
                tqdm.write(f"Plot is saved in '{self._save}'.")

            plt.show() if self._show else plt.close()
        else:
            raise NothingToPlotError


class ShallowWaterPlotter(Plotter[np.ndarray]):
    _benchmark: swe.ShallowWaterBenchmark
    _height_figure: ...
    _height_axes: ...
    _discharge_figure: ...
    _discharge_axes: ...
    _velocity_figure: ...
    _velocity_axes: ...
    _constant_bathymetry: bool

    def __init__(self, benchmark: swe.ShallowWaterBenchmark, **kwargs):
        Plotter.__init__(self, benchmark, **kwargs)

        self._height_figure, self._height_axes = plt.subplots(num="Height Plot")
        self._discharge_figure, self._discharge_axes = plt.subplots(
            num="Discharge Plot"
        )
        self._velocity_figure, self._velocity_axes = plt.subplots(num="Velocity Plot")

        self._build_constant_bathymetry()

    def _build_constant_bathymetry(self):
        bathymetry = swe.build_bathymetry_discretization(
            self._benchmark, len(self._grid)
        )
        self._constant_bathymetry = swe.is_constant(bathymetry)

    def add_initial_data(self):
        self.add_function(self._benchmark.initial_data, "$h_0$", "$q_0$", "$v_0$")

    def add_exact_solution(self):
        end_time = self._benchmark.end_time
        self.add_function(
            self._benchmark.exact_solution_at_end_time,
            f"$h(t={end_time:.1f})$"
            if self._constant_bathymetry
            else f"$h+b(t={end_time:.1f})$",
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
        self._height_axes.plot(grid, total_height, label=label, linewidth=3)

    def add_discharge(self, grid: np.ndarray, values: np.ndarray, label: str):
        discharge = swe.get_discharge(values)
        self._discharge_axes.plot(grid, discharge, label=label, linewidth=3)

    def add_velocity(self, grid: np.ndarray, values: np.ndarray, label: str):
        velocity = swe.get_velocity(values)
        self._velocity_axes.plot(grid, velocity, label=label, linewidth=3)

    def _add_bathymetry(self):
        bathymetry_values = np.array(
            [self._benchmark.bathymetry(x) for x in self._grid]
        )
        self._height_axes.plot(self._grid, bathymetry_values, label="$b$")

    def show(self):
        if self._plot_available:
            self._setup()
            self._save_plots()
            self._show_plots()
        else:
            raise NothingToPlotError

    def _setup(self):
        if not self._constant_bathymetry:
            self._add_bathymetry()

        _, y_max = self._height_axes.get_ylim()
        self._height_axes.set_ylim(bottom=0.0, top=1.05 * y_max, auto=True)

        for ax in [self._height_axes, self._discharge_axes, self._velocity_axes]:
            ax.legend(loc="center left", fontsize="x-large")
            ax.grid()
            ax.tick_params(labelsize="xx-large")

    def _save_plots(self):
        if self._save:
            base_name, extension = os.path.splitext(self._save)

            self._height_figure.savefig(base_name + "_h" + extension)
            self._discharge_figure.savefig(base_name + "_q" + extension)
            self._velocity_figure.savefig(base_name + "_v" + extension)

            tqdm.write(f"Plots are saved in '{base_name}_{{h,q,l}}{extension}'.")

    def _show_plots(self):
        plt.show() if self._show else plt.close("all")


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
        if self._solver is not None and not self._solver_executed:
            self._calculate_solutions()

        self._add_plots()

        try:
            self._plotter.show()
        except NothingToPlotError:
            if self._write_warnings:
                tqdm.write("WARNING: Nothing to plot...")

    def _calculate_solutions(self):
        for solver in tqdm(self._solver, desc="Calculate", unit="solver", leave=False):
            try:
                Calculate(solver).execute()
            except Exception as error:
                if self._write_warnings:
                    tqdm.write(
                        f"WARNING: {str(error)} Solution calculated until t={solver.solution.time:.3e}."
                    )

    def _delete_not_solved_solutions(self):
        accepted_solver = []
        for solver in self._solver:
            if solver.solution.time == self._benchmark.end_time:
                accepted_solver.append(solver)

        self._solver = accepted_solver

    def _add_plots(self):
        self._add_exact_solution()

        if self._solver is not None:
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


class PlotParser(CalculateParser):
    _benchmark_default = "plot"
    _save_default = defaults.PLOT_TARGET

    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "plot",
            help="Calculate and plot solutions.",
            description="Plot benchmarks and computed solutions.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_command_arguments(self, parser):
        self._add_plot_mesh_size(parser)
        self._add_initial_data(parser)
        self._add_save(parser)
        self._add_hide(parser)

    def _add_plot_mesh_size(self, parser):
        parser.add_argument(
            "-m",
            "--mesh-size",
            help="Number of points used for plotting.",
            type=core.positive_int,
            metavar="<size>",
            default=defaults.PLOT_MESH_SIZE,
        )

    def _add_initial_data(self, parser):
        parser.add_argument("--initial", help="Show initial data.", action="store_true")

    def _add_save(self, parser):
        parser.add_argument(
            "--save",
            help=f"Save file in specified direction. (const: {self._save_default})",
            nargs="?",
            const=self._save_default,
            metavar="<file>",
        )

    def _add_hide(self, parser):
        parser.add_argument(
            "--hide",
            help=f"Do not show any figures.",
            action="store_false",
            dest="show",
        )

    def postprocess(self, arguments):
        self._adjust_end_time(arguments)
        self._build_solver(arguments)
        self._build_plotter(arguments)
        arguments.command = Plot

        del arguments.problem

    def _build_plotter(self, arguments):
        plotter = {
            "advection": ScalarPlotter,
            "burgers": ScalarPlotter,
            "swe": ShallowWaterPlotter,
        }

        arguments.plotter = plotter[arguments.problem](
            arguments.benchmark,
            mesh_size=arguments.mesh_size,
            save=arguments.save,
            show=arguments.show,
        )

        del arguments.mesh_size
        del arguments.save
        del arguments.show
