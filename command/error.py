import argparse
from typing import Any, Callable, List, Sequence, Tuple, Type, TypeVar

import core
import defaults
import finite_element
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange

from .calculate import Calculate, CalculateParser
from .command import Command

T = TypeVar("T", float, np.ndarray)


class ErrorCalculator:
    _solver_space: core.SolverSpace
    _norm: core.Norm

    def __init__(self, space: core.SolverSpace, error_norm=None):
        self._space = space
        self._norm = error_norm or core.L2Norm(space.mesh)

    def __call__(
        self,
        exact_solution: core.DiscreteSolution | Callable[[float], T],
        discrete_solution: core.DiscreteSolution,
    ) -> Tuple[float, T]:
        _discrete_solution = self._space.element(discrete_solution.value)

        if isinstance(exact_solution, core.DiscreteSolution):
            _exact_solution = self._space.element(exact_solution.value)
        else:
            _exact_solution = lambda cell_index, x: exact_solution(x)

        return discrete_solution.time, self._norm(
            lambda cell_index, x: _exact_solution(cell_index, x)
            - _discrete_solution(cell_index, x)
        )


class EOCCalculator:
    norm_names: Sequence[str]
    _benchmark: core.Benchmark
    _norms: Sequence[Type[core.Norm]]

    def __init__(self, benchmark: core.Benchmark, norms=None):
        self._benchmark = benchmark
        self._norms = norms or [core.L1Norm, core.L2Norm, core.solver_spaces]
        self.norm_names = [norm.name for norm in self._norms]

    def __call__(
        self,
        solvers: Sequence[core.Solver],
        solver_spaces: Sequence[core.SolverSpace],
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
        self, discrete_solution: core.DiscreteSolution, solver_space: core.SolverSpace
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
        self, solver_space: core.SolverSpace
    ) -> Sequence[ErrorCalculator]:
        norms = self._build_norms(solver_space)
        return [ErrorCalculator(solver_space, norm) for norm in norms]

    def _build_norms(self, solver_space: core.SolverSpace) -> Sequence[core.Norm]:
        norms = list()

        if isinstance(solver_space, finite_element.LagrangeSpace):
            quadrature_degree = solver_space.polynomial_degree + 1
        else:
            quadrature_degree = None

        for norm_type in self._norms:
            if norm_type in [core.L2Norm, core.L1Norm]:
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


class CalculateEOC(Command):
    _benchmark: core.Benchmark
    _solvers: Sequence[Sequence[core.Solver]]
    _solver_spaces: Sequence[Sequence[core.SolverSpace]]

    def __init__(
        self,
        benchmark: core.Benchmark,
        solvers: Sequence[Sequence[core.Solver]],
        solver_spaces: Sequence[Sequence[core.SolverSpace]],
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

        except Exception as error:
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


class CalculateEOCParser(CalculateParser):
    _benchmark_default = "eoc"

    @property
    def _problem_parser_adder(self) -> List[Callable]:
        return [
            self._add_advection_parser,
            self._add_burgers_parser,
        ]

    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "eoc",
            help="Compute Experimental convergence order (EOC).",
            description="Compute experimental order of convergence (EOC).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_command_arguments(self, parser):
        self._add_eoc_mesh_size(parser)
        self._add_refine(parser)

    def _add_eoc_mesh_size(self, parser):
        parser.add_argument(
            "-m",
            "--mesh-size",
            help="""Initial mesh size for eoc calculation. Note, that mesh size
            arguments for solvers are not considered.""",
            type=core.positive_int,
            metavar="<size>",
            default=defaults.EOC_MESH_SIZE,
        )

    def _add_refine(self, parser):
        parser.add_argument(
            "-r",
            "--refine",
            help="Specify number of refinements.",
            type=core.positive_int,
            default=defaults.REFINE_NUMBER,
            metavar="<number>",
        )

    def postprocess(self, arguments):
        self._adjust_end_time(arguments)
        self._build_eoc_solutions(arguments)
        arguments.command = CalculateEOC

        del arguments.problem

    def _build_eoc_solutions(self, arguments):
        arguments.solvers = []
        arguments.solver_spaces = []

        tqdm.write("Calculate Solutions")
        tqdm.write("-------------------")
        tqdm.write(
            f"Initial Mesh Size={arguments.mesh_size}, Refinements={arguments.refine} "
        )
        for solver_arguments in tqdm(
            arguments.solver, desc="Calculate Solutions", unit="solver", leave=False
        ):
            solver_type = solver_arguments.solver
            del solver_arguments.solver
            solvers = []
            solver_spaces = []

            for i in trange(
                arguments.refine, desc=solver_arguments.name, unit="refinement"
            ):
                solver_arguments.mesh_size = 2**i * arguments.mesh_size
                solver = solver_type(arguments.benchmark, **vars(solver_arguments))
                Calculate(solver, leave=False).execute()
                solvers.append(solver)
                solver_spaces.append(solver.solution.space)

            arguments.solvers.append(solvers)
            arguments.solver_spaces.append(solver_spaces)

        del arguments.refine
        del arguments.mesh_size
        del arguments.solver
