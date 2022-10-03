import argparse
import textwrap
from typing import Any, List

from defaults import *

from . import types as parser_type
from .action import AdvectionSolverAction, BurgersSolverAction, EOCAction, PlotAction
from .solver_parser import ADVECTION_SOLVER_PARSERS


class ArgumentParserFEM1D:
    """Parser for command line arguments."""

    _parser = argparse.ArgumentParser(
        prog="cls1",
        description=textwrap.dedent(
            """\
        Explore different PDE-Solver for one-dimension conservation laws.

        """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _current_parser_layer: List[Any] = [_parser]

    def __init__(self):
        self._add_program_parsers(self._parser)

    def _add_program_parsers(self, parser):
        program_parsers = parser.add_subparsers(
            title="Programs",
            dest="program",
            metavar="program",
            required=True,
        )

        self._add_help_parser(program_parsers)
        self._add_test_parser(program_parsers)
        self._add_advection_parser(program_parsers)
        self._add_burgers_parser(program_parsers)

        return program_parsers

    def _add_test_parser(self, parsers):
        test_parser = parsers.add_parser(
            "test",
            help="run unit test",
            description="Task for running unit tests. If no argument is given run all tests.",
        )
        test_parser.add_argument(
            "file", nargs="*", help="run unittest contained in FILE_test.py"
        )

    def _add_help_parser(self, parsers):
        help_parser = parsers.add_parser(
            "help",
            help="display help messages",
            description="Task for displaying different help messages for certain objects",
        )

        help_parser.add_argument(
            "page",
            choices=[*ADVECTION_SOLVER_PARSERS.keys(), "benchmark", "plot", "eoc"],
            help="page which should be displayed in terminal",
        )

    def _add_advection_parser(self, parsers):
        advection_parser = parsers.add_parser(
            "advection",
            help="solve linear Advection",
            description="""Solver liner Advection. Available solvers are 'cg',
            'low_cg' and 'mcl'. For more information use 'help' program.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_command_arguments(advection_parser)
        self._add_mesh_size_argument(advection_parser)
        self._add_benchmark_argument(advection_parser)
        self._add_end_time_argument(advection_parser)
        self._add_profile_argument(advection_parser)
        self._add_solver_argument(advection_parser, AdvectionSolverAction)

    def _add_command_arguments(self, parser):
        task_group = parser.add_mutually_exclusive_group(required=True)
        self._add_plot_argument(task_group)
        self._add_eoc_argument(task_group)

    def _add_plot_argument(self, parser):
        parser.add_argument(
            "-p",
            "--plot",
            help="Plot discrete solution. For more information use 'help' program.",
            nargs="*",
            action=PlotAction,
            metavar="PLOT_ARGS",
        )

    def _add_eoc_argument(self, parser):
        parser.add_argument(
            "-e",
            "--eoc",
            help="Compute eoc. For more information use 'help' program.",
            nargs="*",
            action=EOCAction,
            metavar="EOC_ARGS",
        )

    def _add_mesh_size_argument(self, parser):
        parser.add_argument(
            "-m",
            "--mesh-size",
            help="Number of mesh cells. If not specified use the default size for the chosen task.",
            type=parser_type.positive_int,
            metavar="SIZE",
        )

    def _add_benchmark_argument(self, parser):
        parser.add_argument(
            "-b",
            "--benchmark",
            help="Benchmark for conservation law. If not specified use the default one for the chosen task.",
            type=int,
        )

    def _add_end_time_argument(self, parser):
        parser.add_argument(
            "-T",
            "--end-time",
            type=parser_type.positive_float,
            help="End time used by solvers. If not specified use benchmark's end time.",
        )

    def _add_profile_argument(self, parser):
        parser.add_argument(
            "--profile",
            help="Profile program for optimization purposes. You can specify the number of lines to be printed. Otherwise print %(const)s lines.",
            type=int,
            nargs="?",
            const=30,
            default=0,
            metavar="LINES",
        )

    def _add_solver_argument(self, parser, action):
        parser.add_argument(
            "-s",
            "--solver",
            help="Solver on which the task is applied.",
            nargs="+",
            action=action,
        )

    def _add_burgers_parser(self, program_parsers):
        burgers_parser = program_parsers.add_parser(
            "burgers",
            help="solver Burgers",
            description="""Solve Burgers. Available solvers are 'cg',
            'low_cg' and 'mcl'. For more information use 'help' program.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_command_arguments(burgers_parser)
        self._add_mesh_size_argument(burgers_parser)
        self._add_benchmark_argument(burgers_parser)
        self._add_end_time_argument(burgers_parser)
        self._add_profile_argument(burgers_parser)
        self._add_solver_argument(burgers_parser, BurgersSolverAction)

    def parse_args(self) -> argparse.Namespace:
        return self._parser.parse_args()
