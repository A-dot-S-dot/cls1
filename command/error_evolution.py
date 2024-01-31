import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import core
import defaults
import matplotlib.pyplot as plt
import numpy as np
from benchmark import shallow_water
from tqdm.auto import tqdm, trange

from .animate import Animate, ShallowWaterAnimator
from .calculate import Calculate
from .command import Command
from .plot import Plot, PlotParser, ShallowWaterPlotter


class ErrorEvolutionCalculator:
    _norm: core.Norm

    def __call__(
        self,
        solution: core.DiscreteSolutionWithHistory,
        solution_exact: core.DiscreteSolutionWithHistory,
        norm=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns error evolution between SOLUTION and SOLUTION_EXACT.

        Note, both must have the same spatial dimension.

        """
        self._assert_spatial_dimension_equal(solution, solution_exact)

        space = self._build_space(solution, solution_exact)
        norm = norm or core.L2Norm(space.mesh)
        _time, _solution, _solution_exact = self._create_time_evolution_functions(
            solution, solution_exact, space
        )

        return _time, norm(
            lambda cell_index, x: _solution(cell_index, x)
            - _solution_exact(cell_index, x)
        ) / norm(_solution_exact)

    def _assert_spatial_dimension_equal(
        self,
        solution: core.DiscreteSolutionWithHistory,
        solution_exact: core.DiscreteSolutionWithHistory,
    ):
        assert (
            solution.value_history.shape[1:] == solution_exact.value_history.shape[1:]
        ), "Spatial dimension of SOLUTION and SOLUTION_EXACT must be equal."

    def _build_space(
        self, solution: core.DiscreteSolution, solution_exact: core.DiscreteSolution
    ) -> core.SolverSpace:
        space = solution.space or solution_exact.space

        if space is None:
            raise ValueError(
                "Norm cannot be generated using SOLUTION or SOLUTION_EXACT, since a space must be specified."
            )

        return space

    def _create_time_evolution_functions(
        self,
        solution: core.DiscreteSolutionWithHistory,
        solution_exact: core.DiscreteSolutionWithHistory,
        space: core.SolverSpace,
    ) -> Tuple[np.ndarray, Callable, Callable]:
        time, values, values_exact = self._adjust_time(solution, solution_exact)

        _solutions = [space.element(dof) for dof in values]
        _exact_solutions = [space.element(dof) for dof in values_exact]

        return (
            time,
            lambda cell_index, x: np.array(
                [solution(cell_index, x) for solution in _solutions]
            ),
            lambda cell_index, x: np.array(
                [solution(cell_index, x) for solution in _exact_solutions]
            ),
        )

    def _adjust_time(
        self,
        solution: core.DiscreteSolutionWithHistory,
        solution_exact: core.DiscreteSolutionWithHistory,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(solution.time_history) > len(solution_exact.time_history):
            solution_exact_values = core.TemporalInterpolator()(
                solution_exact.time_history,
                solution_exact.value_history,
                solution.time_history,
            )
            solution_values = solution.value_history
            time = solution.time_history
        else:
            solution_exact_values = solution_exact.value_history
            solution_values = core.TemporalInterpolator()(
                solution.time_history,
                solution.value_history,
                solution_exact.time_history,
            )
            time = solution_exact.time_history

        return time, solution_values, solution_exact_values


class PlotShallowWaterErrorEvolution(Command):
    _solver: List[core.Solver]
    _suptitle: str
    _show: bool
    _save: Optional[str]
    _solver_executed: bool
    _error_evolution_calculator: ErrorEvolutionCalculator

    def __init__(
        self,
        solver: List[core.Solver],
        suptitle=None,
        show=True,
        save=None,
        solver_executed=False,
    ):
        assert len(solver) == 2, "Two solutions are required."

        self._solver = solver
        self._suptitle = suptitle or "Relative $L^2$-Error Evolution"
        self._show = show
        self._save = save
        self._solver_executed = solver_executed

        self._error_evolution_calculator = ErrorEvolutionCalculator()

    def execute(self, time=None, error=None):
        if not self._solver_executed:
            self._calculate_solutions()

        if time is None or error is None:
            time, error = self._error_evolution_calculator(
                self._solver[0].solution, self._solver[1].solution
            )

        self._plot(time, error)

    def _calculate_solutions(self):
        tqdm.write("\nCalculate solutions")
        tqdm.write("-------------------")
        Calculate(self._solver).execute()

    def _plot(self, time: np.ndarray, error: np.ndarray):
        plt.plot(time, error[:, 0], label="Height $L^2$-Error", linewidth=3)
        plt.plot(time, error[:, 1], label="Discharge $L^2$-Error", linewidth=3)
        plt.grid()
        plt.legend(loc="upper right", fontsize="x-large")
        plt.suptitle(self._suptitle)
        plt.tick_params(labelsize="xx-large")

        if self._save:
            plt.savefig(self._save)
            tqdm.write(f"Error plot is saved in '{self._save}'.")

        plt.show() if self._show else plt.close()


class GenerateShallowWaterErrorEvolutionSeries(Command):
    errors: List[np.ndarray]
    times: List[np.ndarray]

    _solver_approximation: core.Solver
    _solver_exact: core.Solver
    _get_benchmark: shallow_water.RandomBenchmarkGenerator
    _seed: Optional[int]
    _initial_conditions: int
    _description: str
    _directory: str
    _save_plot: bool
    _save_animation: bool
    _save_error: bool
    _benchmark_parameters: Dict

    def __init__(
        self,
        solver_approximation: core.Solver,
        solver_exact: core.Solver,
        seed=None,
        end_time=None,
        initial_conditions=20,
        description=None,
        save_directory=None,
        save_plot=False,
        save_animation=False,
        save_error=False,
        **benchmark_parameters,
    ):
        self.errors = []
        self.times = []
        self._solver_approximation = solver_approximation
        self._solver_exact = solver_exact
        self._get_benchmark = shallow_water.RandomBenchmarkGenerator(
            seed=seed, end_time=end_time, **benchmark_parameters
        )
        self._initial_conditions = initial_conditions
        self._description = f"for {description}" or ""
        self._directory = save_directory or ""
        self._save_plot = save_plot
        self._save_animation = save_animation
        self._save_error = save_error

        if save_plot or save_animation or save_error:
            assert self._directory != "", "SAVE_DIRECTORY must be specified"
            os.makedirs(self._directory, exist_ok=True)

    def execute(self):
        for i in trange(
            self._initial_conditions,
            desc="Calculate Evolution Error",
            unit="benchmark",
            leave=False,
        ):
            benchmark = self._get_benchmark()
            self._solver_approximation.reinitialize(benchmark)
            self._solver_exact.reinitialize(benchmark)

            try:
                Calculate(
                    [self._solver_exact, self._solver_approximation], leave=False
                ).execute()
            except Exception as error:
                tqdm.write(f"WARNING: {error} No error plot for seed={benchmark.seed}.")
            else:
                _time, _error = ErrorEvolutionCalculator()(
                    self._solver_approximation.solution, self._solver_exact.solution
                )
                self.times.append(_time)
                self.errors.append(_error)
                self._save(benchmark, i, _time, _error)

    def _save(
        self,
        benchmark: shallow_water.RandomOscillationNoTopographyBenchmark,
        index: int,
        time: np.ndarray,
        error: np.ndarray,
    ):
        solver = [self._solver_approximation, self._solver_exact]

        if self._save_error:
            PlotShallowWaterErrorEvolution(
                solver,
                suptitle=f"Relative $L^2$-Error {self._description} (seed={benchmark.seed})",
                show=False,
                save=f"{self._directory}/error_{index}.png",
                solver_executed=True,
            ).execute(time=time, error=error)

        if self._save_plot:
            plotter = ShallowWaterPlotter(
                benchmark, show=False, save=f"{self._directory}/plot_{index}.png"
            )
            Plot(
                benchmark, solver, plotter, solver_executed=True, write_warnings=False
            ).execute()

        if self._save_animation:
            animator = ShallowWaterAnimator(
                benchmark, show=False, save=f"{self._directory}/animation_{index}.mp4"
            )
            Animate(
                benchmark, solver, animator, solver_executed=True, write_warnings=False
            ).execute()


class PlotShallowWaterAverageErrorEvolution(Command):
    _times: Sequence[np.ndarray]
    _errors: Sequence[np.ndarray]
    _show: bool
    _save: Optional[str]

    def __init__(
        self,
        times: Sequence[np.ndarray],
        errors: Sequence[np.ndarray],
        show=True,
        save=None,
    ):
        self._times = times
        self._errors = errors
        self._show = show
        self._save = save

    def execute(self):
        time, errors = self._adjust_errors()
        mean = np.mean(errors, axis=0)
        error_max = np.max(errors, axis=0)

        height_figure, height_axes = plt.subplots(num="Height L2-Error")
        discharge_figure, discharge_axes = plt.subplots(num="Discharge L2-Error")

        for i, ax in enumerate([height_axes, discharge_axes]):
            ax.plot(time, mean[:, i], linewidth=3)
            ax.fill_between(time, mean[:, i], error_max[:, i], alpha=0.2)
            ax.grid()
            ax.tick_params(labelsize="xx-large")

        if self._save:
            base_name, extension = os.path.splitext(self._save)

            height_figure.savefig(base_name + "_h" + extension)
            discharge_figure.savefig(base_name + "_q" + extension)

            tqdm.write(
                f"Average error plots are saved in '{base_name}_{{h,q}}{extension}'."
            )

        plt.show() if self._show else plt.close("all")

    def _adjust_errors(self) -> Tuple[np.ndarray, np.ndarray]:
        interpolator = core.TemporalInterpolator()
        minimum_time = self._get_minimum_time()
        adjusted_errors = np.array(
            [
                interpolator(time, error, minimum_time)
                for time, error in zip(self._times, self._errors)
            ]
        )

        return minimum_time, adjusted_errors

    def _get_minimum_time(self) -> np.ndarray:
        time_lengths = [len(time) for time in self._times]
        return self._times[np.argmin(time_lengths)]


class PlotErrorEvolutionParser(PlotParser):
    _benchmark_default = "calculate"
    _save_default = defaults.ERROR_EVOLUTION_TARGET

    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "plot-error-evolution",
            help="Plot error between two solutions.",
            description="Plot error between two solutions. Note, two solutions are required. The second one should be the reference solution.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_arguments(self, parser):
        benchmark_group = parser.add_mutually_exclusive_group()
        self._add_benchmark(
            benchmark_group, shallow_water.BENCHMARK_DEFAULTS, shallow_water.BENCHMARKS
        )
        self._add_random_benchmark(benchmark_group)
        self._add_end_time(parser)
        self._add_shallow_water_solver(parser)
        self._add_save(parser)
        self._add_hide(parser)
        self._add_general_arguments(parser)

    def _add_save(self, parser):
        parser.add_argument(
            "--save",
            help=f"Save error evolution plot in specified direction. (const: {self._save_default})",
            nargs="?",
            const=self._save_default,
            metavar="<file>",
        )

    def postprocess(self, arguments):
        self._assert_two_solver(arguments)
        self._adjust_end_time(arguments)
        self._add_save_history_argument(arguments)
        self._build_solver(arguments)
        arguments.command = PlotShallowWaterErrorEvolution

        del arguments.benchmark

    def _assert_two_solver(self, arguments):
        solver_num = 0 if arguments.solver is None else len(arguments.solver)
        assert (
            solver_num == 2
        ), f"Exactly two solver must be given. There are {solver_num}."
