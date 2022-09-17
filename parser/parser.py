import argparse
import textwrap
from typing import Any, List

from defaults import *

from . import types as parser_type
from .solver_action import SolverAction
from .solver_parser import SOLVER_PARSER


class ArgumentParserFEM1D:
    """Parser for command line arguments."""

    _parser = argparse.ArgumentParser(
        prog="cls1",
        description=textwrap.dedent(
            """\
        Explore different PDE-Solver for one-dimension conservation laws.

        The program is structured as follows. First, select a TASK. If you
        choose 'plot' or 'eoc' select a CONSERVATION_LAW to be solved and
        finally the SOLVER(S) to be used. """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _current_parser_layer: List[Any] = [_parser]

    def __init__(self):
        self._add_task_parsers(self._parser)

    def _add_task_parsers(self, parser):
        task_parsers = parser.add_subparsers(
            title="Tasks",
            dest="task",
            metavar="task",
            required=True,
        )

        self._add_help_parser(task_parsers)
        self._add_plot_parser(task_parsers)
        self._add_eoc_parser(task_parsers)
        self._add_test_parser(task_parsers)

        return task_parsers

    def _add_test_parser(self, task_parsers):
        test_parser = task_parsers.add_parser(
            "test",
            help="run unit test",
            description="Task for running unit tests. If no argument is given run all tests.",
        )
        test_parser.add_argument(
            "file", nargs="*", help="run unittest contained in FILE_test.py"
        )

    def _add_help_parser(self, task_parsers):
        help_parser = task_parsers.add_parser(
            "help",
            help="display help messages",
            description="Task for displaying different help messages for certain objects",
        )

        help_parser.add_argument(
            "page",
            choices=[*SOLVER_PARSER.keys(), "benchmark"],
            help="page which should be displayed in terminal",
        )

    def _add_plot_parser(self, task_parsers):
        plot_parser = task_parsers.add_parser(
            "plot",
            help="plot discrete solution",
            description="""Plot solution at a given time and compare with the
            exact solution if it is available. For more information about solver
            and benchmarks use the 'help' command.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        plot_parser.add_argument(
            "--no-plot",
            help="do not display plot (used for test purposes)",
            action="store_true",
        )
        plot_parser.add_argument(
            "-e",
            "--elements-number",
            help="mesh elements number",
            type=parser_type.positive_int,
            metavar="Nh",
            default=ELEMENTS_NUMBER_PLOT,
        )

        self._add_problem_argument(plot_parser, default_benchmark="plot_default")
        plot_parser.add_argument(
            "solver",
            help="solver on which the task is applied. If no solver is given plot the benchmark.",
            nargs="*",
            action=SolverAction,
        )
        self._add_mesh_arguments(plot_parser)
        self._add_profile_argument(plot_parser)

    def _add_problem_argument(self, parser, default_benchmark: str):
        parser.add_argument(
            "problem",
            choices=["advection", "burgers"],
            help="conservation law to be solved",
        )
        parser.add_argument(
            "-b",
            "--benchmark",
            default=default_benchmark,
            help="benchmark for conservation law",
        )

    def _add_mesh_arguments(self, parser):
        mesh_argument_group = parser.add_argument_group("mesh arguments")
        mesh_argument_group.add_argument(
            "--courant-factor",
            help="specify the factor for the number of time steps depending on the number of simplices in the used mesh",
            type=parser_type.positive_int,
            metavar="factor",
            default=COURANT_FACTOR,
        )
        mesh_argument_group.add_argument(
            "-T",
            "--end-time",
            type=parser_type.positive_float,
            help="End time used by solvers. If not specified use benchmark's end time.",
        )

    def _add_profile_argument(self, parser):
        parser.add_argument(
            "-p",
            "--profile",
            help="Profile program for optimization purposes. You can specify the number of lines to be printed. Otherwise print %(const)s lines.",
            type=int,
            nargs="?",
            const=30,
            default=0,
            metavar="lines",
        )

    def _add_eoc_parser(self, task_parsers):
        eoc_parser = task_parsers.add_parser(
            "eoc",
            help="test order of convergence",
            description="""Specify experimental order of convergence (EOC) by
            calculating discrete solutions after several refinements and
            comparing L1-, L2- and Linf-errors.For more information about solver
            and benchmarks use the 'help' command""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        eoc_parser.add_argument(
            "--refine",
            help="specify how many times the mesh should be refined",
            type=parser_type.positive_int,
            default=REFINE_NUMBER,
        )
        eoc_parser.add_argument(
            "-e",
            "--elements-number",
            help="initial mesh elements number",
            type=parser_type.positive_int,
            metavar="Nh",
            default=ELEMENTS_NUMBER_EOC,
        )

        self._add_problem_argument(eoc_parser, default_benchmark="eoc_default")
        eoc_parser.add_argument(
            "solver",
            help="solver on which the task is applied.",
            nargs="+",
            action=SolverAction,
        )
        self._add_mesh_arguments(eoc_parser)
        self._add_profile_argument(eoc_parser)

    def parse_args(self) -> argparse.Namespace:
        return self._parser.parse_args()
