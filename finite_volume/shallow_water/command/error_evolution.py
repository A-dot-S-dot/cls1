import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import core
import defaults
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm, trange

import command as cmd


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
        space = solution._space or solution_exact._space

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


class PlotShallowWaterErrorEvolution(cmd.Command):
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
        cmd.Calculate(self._solver).execute()

    def _plot(self, time: np.ndarray, error: np.ndarray):
        plt.plot(time, error[:, 0], label="height error")
        plt.plot(time, error[:, 1], label="discharge error")
        plt.suptitle(self._suptitle)
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("relative error")

        if self._save:
            plt.savefig(self._save)
            tqdm.write(f"Error plot is saved in '{self._save}'.")

        plt.show() if self._show else plt.close()


# class GenerateShallowWaterErrorEvolutionSeries(cmd.Command):
#     """Plot Error Evolution for considered benchmark parameters."""

#     errors: List[np.ndarray]
#     times: List[np.ndarray]

#     _seed: int
#     _initial_conditions: int
#     _description: str
#     _directory: str
#     _save_plot: bool
#     _save_animation: bool
#     _save_error: bool
#     _benchmark_parameters: Dict

#     def __init__(
#         self,
#         times: List[np.ndarray],
#         errors: List[np.ndarray],
#         seed=0,
#         initial_conditions=20,
#         description=None,
#         save_directory=None,
#         save_plot=False,
#         save_animation=False,
#         save_error=False,
#         **benchmark_parameters,
#     ):
#         self.times = times
#         self.errors = errors

#         self._seed = seed
#         self._initial_conditions = initial_conditions
#         self._description = f"for {description}" or ""
#         self._directory = save_directory or ""
#         self._save_plot = save_plot
#         self._save_animation = save_animation
#         self._save_error = save_error
#         self._benchmark_parameters = benchmark_parameters

#         if save_plot or save_animation or save_error:
#             assert self._directory != "", "SAVE_DIRECTORY must be specified"
#             os.makedirs(self._directory, exist_ok=True)

#     def execute(self):
#         for i in trange(
#             self._initial_conditions,
#             desc="Calculate Evolution Error",
#             unit="benchmark",
#             leave=False,
#         ):
#             benchmark = swe.RandomOscillationNoTopographyBenchmark(
#                 seed=self._seed + i, **self._benchmark_parameters
#             )
#             approximated_solver = solver.SubgridNetworkSolver(
#                 benchmark, save_history=True
#             )
#             exact_solver = sovler.CoarseLowOrderSolver(benchmark, save_history=True)

#             try:
#                 cmd.Calculate(
#                     [exact_solver, approximated_solver], leave=False
#                 ).execute()
#             except Exception as error:
#                 tqdm.write(f"WARNING: {error} No error plot for seed={i}.")
#             else:
#                 _time, _error = ErrorEvolutionCalculator()(
#                     exact_solver.solution, approximated_solver.solution
#                 )
#                 self.times.append(_time)
#                 self.errors.append(_error)
#                 self._save(
#                     benchmark, i, approximated_solver, exact_solver, _time, _error
#                 )

#     def _save(
#         self,
#         benchmark: swe.ShallowWaterBenchmark,
#         index: int,
#         approximated_solver: core.Solver,
#         exact_solver: core.Solver,
#         time: np.ndarray,
#         error: np.ndarray,
#     ):
#         solver = [approximated_solver, exact_solver]

#         if self._save_error:
#             PlotShallowWaterErrorEvolution(
#                 solver,
#                 f"Relative $L^2$-Error {self._description} (seed={index})",
#                 show=False,
#                 save=f"{self._directory}/error_{index}.png",
#                 solver_executed=True,
#             ).execute(time=time, error=error)

#         if self._save_plot:
#             plotter = cmd.ShallowWaterPlotter(
#                 benchmark, show=False, save=f"{self._directory}/plot_{index}.png"
#             )
#             cmd.Plot(
#                 benchmark, solver, plotter, solver_executed=True, write_warnings=False
#             ).execute()

#         if self._save_animation:
#             animator = cmd.ShallowWaterAnimator(
#                 benchmark, show=False, save=f"{self._directory}/animation_{index}.mp4"
#             )
#             cmd.Animate(
#                 benchmark, solver, animator, solver_executed=True, write_warnings=False
#             ).execute()


# class PlotShallowWaterAverageErrorEvolution(cmd.Command):
#     _times: Sequence[np.ndarray]
#     _errors: Sequence[np.ndarray]
#     _suptitle: Optional[str]
#     _show: bool
#     _save: Optional[str]

#     def __init__(
#         self,
#         times: Sequence[np.ndarray],
#         errors: Sequence[np.ndarray],
#         suptitle=None,
#         show=True,
#         save=None,
#     ):
#         self._times = times
#         self._errors = errors
#         self._suptitle = suptitle or "$L^2$-Error between network and real solution"
#         self._show = show
#         self._save = save

#     def execute(self):
#         time, errors = self._adjust_errors()
#         mean = np.mean(errors, axis=0)
#         error_min = np.min(errors, axis=0)
#         error_max = np.max(errors, axis=0)

#         fig, (height_ax, discharge_ax) = plt.subplots(1, 2)

#         height_ax.plot(time, mean[:, 0], label="height error")
#         height_ax.fill_between(time, error_min[:, 0], error_max[:, 0], alpha=0.2)
#         height_ax.legend()
#         height_ax.set_xlabel("time")
#         height_ax.set_ylabel("error")

#         discharge_ax.plot(time, mean[:, 1], label="discharge error")
#         discharge_ax.fill_between(time, error_min[:, 1], error_max[:, 1], alpha=0.2)
#         discharge_ax.legend()
#         discharge_ax.set_xlabel("time")
#         discharge_ax.set_ylabel("error")

#         fig.suptitle(self._suptitle)

#         if self._save:
#             fig.savefig(self._save)

#         plt.show() if self._show else plt.close()

#     def _adjust_errors(self) -> Tuple[np.ndarray, np.ndarray]:
#         interpolator = core.TemporalInterpolator()
#         minimum_time = self._get_minimum_time()
#         adjusted_errors = np.array(
#             [
#                 interpolator(time, error, minimum_time)
#                 for time, error in zip(self._times, self._errors)
#             ]
#         )

#         return minimum_time, adjusted_errors

#     def _get_minimum_time(self) -> np.ndarray:
#         time_lengths = [len(time) for time in self._times]
#         return self._times[np.argmin(time_lengths)]
