from typing import Callable, Optional, Sequence, Tuple, Type, TypeVar

import benchmark
import defaults
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pde_solver.norm as norm
from benchmark.abstract import NoExactSolutionError
from pde_solver.discretization import DiscreteSolution, SolverSpace
from pde_solver.discretization.discrete_solution import TemporalInterpolation
from pde_solver.discretization.finite_element import (
    LagrangeFiniteElementSpace,
)
from pde_solver.solver import Solver
from tqdm import tqdm

from .command import Command

T = TypeVar("T", float, np.ndarray)


class ErrorCalculator:
    _solver_space: SolverSpace
    _norm: norm.Norm

    def __init__(self, space: SolverSpace, error_norm: norm.Norm):
        self._space = space
        self._norm = error_norm

    def __call__(
        self,
        exact_solution: DiscreteSolution | Callable[[float], T],
        discrete_solution: DiscreteSolution,
    ) -> Tuple[float, T]:
        _discrete_solution = self._space.element(discrete_solution.end_values)

        if isinstance(exact_solution, DiscreteSolution):
            _exact_solution = self._space.element(exact_solution.end_values)
        else:
            _exact_solution = lambda cell_index, x: exact_solution(x)

        return discrete_solution.end_time, self._norm(
            lambda cell_index, x: _exact_solution(cell_index, x)
            - _discrete_solution(cell_index, x)
        )


class TimeEvolutionErrorCalculator:
    _solver_space: SolverSpace
    _norm: norm.Norm

    def __init__(self, space: SolverSpace, error_norm: norm.Norm):
        self._space = space
        self._norm = error_norm

    def __call__(
        self,
        exact_solution: DiscreteSolution | Callable[[float, float], T],
        discrete_solution: DiscreteSolution,
    ):
        if isinstance(exact_solution, DiscreteSolution):
            (
                _time,
                _exact_solution,
                _discrete_solution,
            ) = self._create_time_evolution_functions(exact_solution, discrete_solution)
        else:
            _exact_solution = lambda cell_index, x: np.array(
                [exact_solution(x, t) for t in discrete_solution.time]
            )
            _discrete_solutions = [
                self._space.element(values) for values in discrete_solution.values
            ]
            _discrete_solution = lambda cell_index, x: np.array(
                [solution(cell_index, x) for solution in _discrete_solutions]
            )
            _time = discrete_solution.time

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
        if len(discrete_solution.time) > len(exact_solution.time):
            exact_solution_values = TemporalInterpolation()(
                exact_solution, discrete_solution.time
            )
            discrete_solution_values = discrete_solution.values
            time = discrete_solution.time
        else:
            exact_solution_values = exact_solution.values
            discrete_solution_values = TemporalInterpolation()(
                discrete_solution, exact_solution.time
            )
            time = exact_solution.time

        return time, exact_solution_values, discrete_solution_values


class EOCCalculator:
    norm_names: Sequence[str]
    _benchmark: benchmark.Benchmark
    _norms: Sequence[Type[norm.Norm]]

    def __init__(self, benchmark: benchmark.Benchmark, norms=None):
        self._benchmark = benchmark
        self._norms = norms or [norm.L1Norm, norm.L2Norm, norm.solver_spaces]
        self.norm_names = [norm.name for norm in self._norms]

    def __call__(
        self,
        solvers: Sequence[Solver],
        solver_spaces: Sequence[SolverSpace],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        refine_number = len(solvers) - 1
        dofs = np.empty(refine_number + 1)
        errors = np.empty((3, refine_number + 1))
        eocs = np.empty((3, refine_number + 1))

        for index, (solver, solver_space) in enumerate(zip(solvers, solver_spaces)):
            dofs[index] = solver.solution.dimension
            errors[:, index] = self._calculate_error(solver.solution, solver_space)
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

        if isinstance(solver_space, LagrangeFiniteElementSpace):
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
    _error_norm: norm.Norm
    _save: Optional[str]

    def __init__(
        self,
        space: SolverSpace,
        exact_solution: DiscreteSolution,
        discrete_solution: DiscreteSolution,
        error_norm=None,
        save=None,
    ):
        self._error_norm = error_norm or norm.L2Norm(space.mesh, 2)
        self._time, self._error = TimeEvolutionErrorCalculator(space, self._error_norm)(
            exact_solution, discrete_solution
        )
        self._save = save or defaults.ERROR_EVOLUTION_PATH

    def execute(self):
        plt.plot(self._time, self._error[:, 0], label="height error")
        plt.plot(self._time, self._error[:, 1], label="discharge error")
        plt.suptitle(f"{self._error_norm.name}-Error between network and real solution")
        plt.legend()
        plt.xlabel("time")
        plt.ylabel("error")

        if self._save:
            plt.savefig(self._save)

        plt.show()


class CalculateEOC(Command):
    _benchmark: benchmark.Benchmark
    _solvers: Sequence[Sequence[Solver]]
    _solver_spaces: Sequence[Sequence[SolverSpace]]

    def __init__(
        self,
        benchmark: benchmark.Benchmark,
        solvers: Sequence[Sequence[Solver]],
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
