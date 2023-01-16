import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

import defaults
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from base import norm, solver
from base.benchmark import Benchmark, NoExactSolutionError
from base.discretization import DiscreteSolution, SolverSpace
from base.discretization.finite_element import LagrangeSpace
from base.interpolate import TemporalInterpolator
from tqdm import tqdm

from .command import Command

T = TypeVar("T", float, np.ndarray)


class ErrorCalculator:
    _solver_space: SolverSpace
    _norm: norm.Norm

    def __init__(self, space: SolverSpace, error_norm=None):
        self._space = space
        self._norm = error_norm or norm.L2Norm(space.mesh)

    def __call__(
        self,
        exact_solution: DiscreteSolution | Callable[[float], T],
        discrete_solution: DiscreteSolution,
    ) -> Tuple[float, T]:
        _discrete_solution = self._space.element(discrete_solution.value)

        if isinstance(exact_solution, DiscreteSolution):
            _exact_solution = self._space.element(exact_solution.value)
        else:
            _exact_solution = lambda cell_index, x: exact_solution(x)

        return discrete_solution.time, self._norm(
            lambda cell_index, x: _exact_solution(cell_index, x)
            - _discrete_solution(cell_index, x)
        )


class ErrorEvolutionCalculator(ErrorCalculator):
    _solver_space: SolverSpace
    _norm: norm.Norm

    def __call__(
        self,
        exact_solution: DiscreteSolution | Callable[[float, float], float | np.ndarray],
        discrete_solution: DiscreteSolution,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(exact_solution, DiscreteSolution):
            (
                _time,
                _exact_solution,
                _discrete_solution,
            ) = self._create_time_evolution_functions(exact_solution, discrete_solution)
        else:
            _exact_solution = lambda cell_index, x: np.array(
                [exact_solution(x, t) for t in discrete_solution.time_history]
            )
            _discrete_solutions = [
                self._space.element(values) for values in discrete_solution.value
            ]
            _discrete_solution = lambda cell_index, x: np.array(
                [solution(cell_index, x) for solution in _discrete_solutions]
            )
            _time = discrete_solution.time_history

        return _time, self._norm(
            lambda cell_index, x: _exact_solution(cell_index, x)
            - _discrete_solution(cell_index, x)
        )

    def _create_time_evolution_functions(
        self, exact_solution: DiscreteSolution, discrete_solution: DiscreteSolution
    ) -> Tuple[np.ndarray, Callable, Callable]:
        time, exact_values, discrete_values = self._adjust_time(
            exact_solution, discrete_solution
        )

        _exact_solutions = [self._space.element(values) for values in exact_values]
        _discrete_solutions = [
            self._space.element(values) for values in discrete_values
        ]

        return (
            time,
            lambda cell_index, x: np.array(
                [solution(cell_index, x) for solution in _exact_solutions]
            ),
            lambda cell_index, x: np.array(
                [solution(cell_index, x) for solution in _discrete_solutions]
            ),
        )

    def _adjust_time(
        self, exact_solution: DiscreteSolution, discrete_solution: DiscreteSolution
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(discrete_solution.time_history) > len(exact_solution.time_history):
            exact_solution_values = TemporalInterpolator()(
                exact_solution.time_history,
                exact_solution.value,
                discrete_solution.time_history,
            )
            discrete_solution_values = discrete_solution.value
            time = discrete_solution.time_history
        else:
            exact_solution_values = exact_solution.value
            discrete_solution_values = TemporalInterpolator()(
                discrete_solution.time_history,
                discrete_solution.value,
                exact_solution.time_history,
            )
            time = exact_solution.time_history

        return time, exact_solution_values, discrete_solution_values


class EOCCalculator:
    norm_names: Sequence[str]
    _benchmark: Benchmark
    _norms: Sequence[Type[norm.Norm]]

    def __init__(self, benchmark: Benchmark, norms=None):
        self._benchmark = benchmark
        self._norms = norms or [norm.L1Norm, norm.L2Norm, norm.solver_spaces]
        self.norm_names = [norm.name for norm in self._norms]

    def __call__(
        self,
        solvers: Sequence[solver.Solver],
        solver_spaces: Sequence[SolverSpace],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        refine_number = len(solvers) - 1
        dofs = np.empty(refine_number + 1)
        errors = np.empty((3, refine_number + 1))
        eocs = np.empty((3, refine_number + 1))

        for index, (solver, solver_space) in enumerate(zip(solvers, solver_spaces)):
            dofs[index] = solver._solution.dimension
            errors[:, index] = self._calculate_error(solver._solution, solver_space)
            eocs[:, index] = self._calculate_eoc(errors, index)

        return dofs, errors, eocs

    def _calculate_error(
        self, discrete_solution: DiscreteSolution, solver_space: SolverSpace
    ) -> np.ndarray:
        exact_solution = self._benchmark.exact_solution_at_end_time
        error_calculators = self._build_error_calculators(solver_space)

        return np.array(
            [
                error_calculator(exact_solution, discrete_solution)[1]
                for error_calculator in error_calculators
            ]
        )

    def _build_error_calculators(
        self, solver_space: SolverSpace
    ) -> Sequence[ErrorCalculator]:
        norms = self._build_norms(solver_space)
        return [ErrorCalculator(solver_space, norm) for norm in norms]

    def _build_norms(self, solver_space: SolverSpace) -> Sequence[norm.Norm]:
        norms = list()

        if isinstance(solver_space, LagrangeSpace):
            quadrature_degree = solver_space.polynomial_degree + 1
        else:
            quadrature_degree = None

        for norm_type in self._norms:
            if norm_type in [norm.L2Norm, norm.L1Norm]:
                norms.append(
                    norm_type(solver_space.mesh, quadrature_degree=quadrature_degree)
                )
            else:
                norms.append(norm_type(solver_space.mesh))

        return norms

    def _calculate_eoc(self, errors: np.ndarray, current_index: int) -> np.ndarray:
        if current_index == 0:
            return np.array([np.nan, np.nan, np.nan])
        else:
            return np.array(
                [
                    np.log2(old_error / new_error)
                    for old_error, new_error in zip(
                        errors[:, current_index - 1], errors[:, current_index]
                    )
                ]
            )


class EOCDataFrame:
    dofs_format = "{:.0f}"
    error_format = "{:.2e}"
    eoc_format = "{:.2f}"

    _norm_names: Sequence[str]
    _data_frame: pd.DataFrame

    def __call__(
        self,
        dofs: np.ndarray,
        errors: np.ndarray,
        eocs: np.ndarray,
        norm_names: Sequence[str],
    ) -> pd.DataFrame:
        self._norm_names = norm_names
        self._create_empty_data_frame(len(dofs))
        self._fill_data_frame(dofs, errors, eocs)
        self._format_data_frame()

        return self._data_frame

    def _create_empty_data_frame(self, row_number: int):
        columns = pd.MultiIndex.from_product([self._norm_names, ["error", "eoc"]])
        columns = pd.MultiIndex.from_tuples([("DOFs", ""), *columns])
        index = pd.Index(
            [i for i in range(row_number)],
            name="refinement",
        )

        self._data_frame = pd.DataFrame(columns=columns, index=index)

    def _fill_data_frame(self, dofs: np.ndarray, errors: np.ndarray, eocs: np.ndarray):
        self._data_frame["DOFs"] = dofs

        for norm_index, norm in enumerate(self._norm_names):
            self._data_frame[norm, "error"] = errors[norm_index]
            self._data_frame[norm, "eoc"] = eocs[norm_index]

    def _format_data_frame(self):
        self._data_frame["DOFs"] = self._data_frame["DOFs"].apply(
            self.dofs_format.format
        )

        for norm in self._norm_names:
            self._data_frame[norm, "error"] = self._data_frame[norm, "error"].apply(
                self.error_format.format
            )
            self._data_frame[norm, "eoc"] = self._data_frame[norm, "eoc"].apply(
                self.eoc_format.format
            )

    def __repr__(self) -> str:
        return self._data_frame.__repr__()


class PlotShallowWaterErrorEvolution(Command):
    _time: np.ndarray
    _error: np.ndarray
    _suptitle: str
    _show: bool
    _save: Optional[str]

    def __init__(
        self,
        time: np.ndarray,
        error: np.ndarray,
        suptitle=None,
        show=True,
        save=None,
    ):
        self._time = time
        self._error = error
        self._suptitle = suptitle or "Error between network and real solution"
        self._show = show
        self._save = save or defaults.ERROR_EVOLUTION_PATH

    def execute(self):
        plt.plot(self._time, self._error[:, 0], label="height error")
        plt.plot(self._time, self._error[:, 1], label="discharge error")
        plt.suptitle(self._suptitle)
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("error")

        if self._save:
            plt.savefig(self._save)

        plt.show() if self._show else plt.close()


class CalculateEOC(Command):
    _benchmark: Benchmark
    _solvers: Sequence[Sequence[solver.Solver]]
    _solver_spaces: Sequence[Sequence[SolverSpace]]

    def __init__(
        self,
        benchmark: Benchmark,
        solvers: Sequence[Sequence[solver.Solver]],
        solver_spaces: Sequence[Sequence[SolverSpace]],
    ):
        self._benchmark = benchmark
        self._solvers = solvers
        self._solver_spaces = solver_spaces

    def execute(self):
        if len(self._solvers) == 0:
            print("WARNING: Nothing to do...")
            return

        try:
            self._print_eocs()

        except NoExactSolutionError as error:
            print("ERROR: " + str(error))

    def _print_eocs(self):
        eocs = []
        titles = []

        for solvers, solver_spaces in tqdm(
            zip(self._solvers, self._solver_spaces),
            desc="Calculate EOC",
            unit="solver",
            leave=False,
        ):
            titles.append(solvers[0].name)
            eoc_calculator = EOCCalculator(self._benchmark)
            eocs.append(
                EOCDataFrame()(
                    *eoc_calculator(solvers, solver_spaces), eoc_calculator.norm_names
                )
            )

        for eoc, title in zip(eocs, titles):
            print()
            print(title)
            print(len(title) * "-")
            print(eoc)


# class GenerateShallowWaterErrorEvolutionSeries(Command):
#     """Plot Error Evolution for considered benchmark parameters."""

#     errors: List[np.ndarray]
#     times: List[np.ndarray]

#     _directory: str
#     _initial_conditions: int
#     _description: str
#     _save: Optional[str]
#     _benchmark_parameters: Dict

#     def __init__(
#         self,
#         times: List[np.ndarray],
#         errors: List[np.ndarray],
#         initial_conditions=20,
#         description=None,
#         save=None,
#         **benchmark_parameters,
#     ):
#         self.times = times
#         self.errors = errors

#         self._initial_conditions = initial_conditions
#         self._description = f"for {description}" or ""
#         self._save = save
#         self._benchmark_parameters = benchmark_parameters

#         if save:
#             os.makedirs(save, exist_ok=True)

#     def execute(self):
#         for i in trange(
#             self._initial_conditions, desc="Calculate Evolution Error", unit="benchmark"
#         ):
#             title = f"Generate Error {i}"
#             title += "\n" + len(title) * "-"
#             tqdm.write(title)

#             swe_benchmark = (
#                 benchmark.ShallowWaterRandomOscillationNoTopographyBenchmark(
#                     seed=i, **self._benchmark_parameters
#                 )
#             )
#             approximated_solver = solver.ReducedNetworkSolver("swe", swe_benchmark)
#             exact_solver = solver.ReducedExactSolver("swe", swe_benchmark)

#             Calculate([exact_solver, approximated_solver]).execute()
#             tqdm.write("")

#             _time, _error = ErrorEvolutionCalculator(exact_solver.space)(
#                 exact_solver._solution, approximated_solver._solution
#             )
#             self.times.append(_time)
#             self.errors.append(_error)

#             if self._save:
#                 PlotShallowWaterErrorEvolution(
#                     _time,
#                     _error,
#                     f"$L^2$-Error {self._description} (seed={i})",
#                     show=False,
#                     save=f"{self._save}/{i}.png",
#                 ).execute()


class PlotShallowWaterAverageErrorEvolution(Command):
    _times: Sequence[np.ndarray]
    _errors: Sequence[np.ndarray]
    _suptitle: Optional[str]
    _show: bool
    _save: Optional[str]

    def __init__(
        self,
        times: Sequence[np.ndarray],
        errors: Sequence[np.ndarray],
        suptitle=None,
        show=True,
        save=None,
    ):
        self._times = times
        self._errors = errors
        self._suptitle = suptitle or "$L^2$-Error between network and real solution"
        self._show = show
        self._save = save

    def execute(self):
        time, errors = self._adjust_errors()
        mean = np.mean(errors, axis=0)
        error_min = np.min(errors, axis=0)
        error_max = np.max(errors, axis=0)

        fig, (height_ax, discharge_ax) = plt.subplots(1, 2)

        height_ax.plot(time, mean[:, 0], label="height error")
        height_ax.fill_between(time, error_min[:, 0], error_max[:, 0], alpha=0.2)
        height_ax.legend()
        height_ax.set_xlabel("time")
        height_ax.set_ylabel("error")

        discharge_ax.plot(time, mean[:, 1], label="discharge error")
        discharge_ax.fill_between(time, error_min[:, 1], error_max[:, 1], alpha=0.2)
        discharge_ax.legend()
        discharge_ax.set_xlabel("time")
        discharge_ax.set_ylabel("error")

        fig.suptitle(self._suptitle)

        if self._save:
            fig.savefig(self._save)

        plt.show() if self._show else plt.close()

    def _adjust_errors(self) -> Tuple[np.ndarray, np.ndarray]:
        interpolator = TemporalInterpolator()
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
