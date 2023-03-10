import argparse
from typing import Any, Callable, Dict, List, Sequence

import core
import finite_element.scalar.solver as fem_scalar
import finite_volume.scalar.solver as fv_scalar
import finite_volume.shallow_water.solver as fv_swe
from benchmark import advection, burgers, shallow_water
from tqdm.auto import tqdm

from .command import Command, CommandParser

SCALAR_SOLVER_PARSER = fem_scalar.SOLVER_PARSER | fv_scalar.SOLVER_PARSER
SHALLOW_WATER_SOLVER_PARSER = fv_swe.SOLVER_PARSER


class ScalarSolverAction(core.SolverAction):
    solver_parsers = SCALAR_SOLVER_PARSER


class ShallowWaterSolverAction(core.SolverAction):
    solver_parsers = SHALLOW_WATER_SOLVER_PARSER


class Calculate(Command):
    """Calculate discrete solution without doing with it something."""

    _solver: core.Solver | Sequence[core.Solver]
    _tqdm_kwargs: Dict

    def __init__(self, solver: core.Solver | Sequence[core.Solver], **tqdm_kwargs):
        self._solver = solver
        self._tqdm_kwargs = tqdm_kwargs

    def execute(self):
        if isinstance(self._solver, core.Solver):
            self._solver.solve(**self._tqdm_kwargs)
        elif len(self._solver) == 1:
            self._solver[0].solve(**self._tqdm_kwargs)
        else:
            for solver in tqdm(
                self._solver, desc="Calculate", unit="solver", leave=False
            ):
                solver.solve(**self._tqdm_kwargs)


class CalculateParser(CommandParser):
    _benchmark_default = "calculate"

    @property
    def _problem_parser_adder(self) -> List[Callable]:
        return [
            self._add_advection_parser,
            self._add_burgers_parser,
            self._add_shallow_water_parser,
        ]

    def add_parser(self, parsers):
        parser = self._get_parser(parsers)
        self._add_arguments(parser)

    def _get_parser(self, parsers) -> Any:
        return parsers.add_parser(
            "calculate",
            help="Calculate solutions without doing with them something.",
            description="Calculate solutions without doing with them something.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    def _add_arguments(self, parser):
        problem_parsers = parser.add_subparsers(
            title="Problems", dest="problem", metavar="<problem>", required=True
        )

        for add_problem_parser in self._problem_parser_adder:
            problem_parser = add_problem_parser(problem_parsers)
            self._add_command_arguments(problem_parser)
            self._add_general_arguments(problem_parser)

    def _add_command_arguments(self, parser):
        ...

    def _add_advection_parser(self, parsers):
        parser = parsers.add_parser(
            "advection",
            help="Solve linear Advection.",
            description="""Solve liner Advection. For more information use 'cls1
            help SOLVER'.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_benchmark(parser, advection.BENCHMARK_DEFAULTS, advection.BENCHMARKS)
        self._add_end_time(parser)
        self._add_scalar_solver(parser)

        return parser

    def _add_benchmark(self, parser, benchmark_defaults, benchmarks):
        parser.add_argument(
            "-b",
            "--benchmark",
            help="""Choose benchmark by key. Available keys are: """
            + ", ".join([*benchmarks.keys()]),
            type=lambda input: benchmarks[input](),
            metavar="<name>",
            default=benchmark_defaults[self._benchmark_default](),
        )

    def _add_end_time(self, parser):
        parser.add_argument(
            "-T",
            "--end-time",
            type=core.positive_float,
            help="End time used by solvers. If not specified use benchmark's end time.",
            metavar="<time>",
        )

    def _add_scalar_solver(self, parser):
        self._add_solver(parser, SCALAR_SOLVER_PARSER.keys(), ScalarSolverAction)

    def _add_solver(self, parser, solver, action):
        parser.add_argument(
            "-s",
            "--solver",
            help="Solver on which the task is applied. Available solver are: "
            + ", ".join([*solver]),
            nargs="+",
            action=action,
            metavar="<solver> [options]",
        )

    def _add_burgers_parser(self, parsers):
        parser = parsers.add_parser(
            "burgers",
            help="Solve Burgers.",
            description="""Solve Burgers. For more information use 'cls1
            help SOLVER'.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_benchmark(parser, burgers.BENCHMARK_DEFAULTS, burgers.BENCHMARKS)
        self._add_end_time(parser)
        self._add_scalar_solver(parser)

        return parser

    def _add_shallow_water_parser(self, parsers):
        parser = parsers.add_parser(
            "swe",
            help="Solve Shallow Water equations.",
            description="""Solve shallow water equations (SWE). For more
            information use 'cls1 help SOLVER'.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        benchmark_group = parser.add_mutually_exclusive_group()
        self._add_benchmark(
            benchmark_group, shallow_water.BENCHMARK_DEFAULTS, shallow_water.BENCHMARKS
        )
        self._add_random_benchmark(benchmark_group)
        self._add_end_time(parser)
        self._add_shallow_water_solver(parser)

        return parser

    def _add_shallow_water_solver(self, parser):
        self._add_solver(
            parser, SHALLOW_WATER_SOLVER_PARSER.keys(), ShallowWaterSolverAction
        )

    def _add_random_benchmark(self, parser):
        parser.add_argument(
            "-r",
            "--random-benchmark",
            help="Choose random benchmark.",
            type=lambda input: shallow_water.RandomOscillationNoTopographyBenchmark(
                seed=int(input)
            ),
            nargs="?",
            const=shallow_water.RandomOscillationNoTopographyBenchmark(),
            metavar="<seed>",
            dest="benchmark",
        )

    def postprocess(self, arguments):
        self._adjust_end_time(arguments)
        self._build_solver(arguments)
        arguments.command = Calculate

        del arguments.problem
        del arguments.benchmark

    def _adjust_end_time(self, arguments):
        if arguments.end_time:
            arguments.benchmark.end_time = arguments.end_time
        del arguments.end_time

    def _build_solver(self, arguments):
        solver_list = []

        for solver_arguments in arguments.solver:
            solver = solver_arguments.solver
            del solver_arguments.solver

            solver_list.append(solver(arguments.benchmark, **vars(solver_arguments)))

        arguments.solver = solver_list
