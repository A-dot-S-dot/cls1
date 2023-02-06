import argparse
import textwrap

import command as cmd
import defaults
from scalar.benchmark import advection, burgers
from shallow_water import benchmark as shallow_water

from . import action, argument
from . import postprocessing as ppr
from .solver_parser import SOLVER_PARSERS


class CustomArgumentParser:
    """Parser for command line arguments."""

    _arguments_postprocessing = {
        "test": [ppr.BuildCommand(cmd.Test)],
        "help": [ppr.BuildCommand(cmd.Help)],
        "calculate": [
            ppr.adjust_end_time,
            ppr.build_solver,
            ppr.BuildCommand(cmd.Calculate),
            ppr.DeleteArguments("problem", "benchmark"),
        ],
        "plot": [
            ppr.adjust_end_time,
            ppr.build_solver,
            ppr.build_plotter,
            ppr.BuildCommand(cmd.Plot),
            ppr.DeleteArguments("problem"),
        ],
        "animate": [
            ppr.adjust_end_time,
            ppr.add_save_history_argument,
            ppr.build_solver,
            ppr.build_animator,
            ppr.BuildCommand(cmd.Animate),
            ppr.DeleteArguments("problem"),
        ],
        "eoc": [
            ppr.adjust_end_time,
            ppr.build_eoc_solutions,
            ppr.BuildCommand(cmd.CalculateEOC),
            ppr.DeleteArguments("problem"),
        ],
        "plot-error-evolution": [
            ppr.adjust_end_time,
            ppr.add_save_history_argument,
            ppr.build_solver,
            ppr.BuildCommand(cmd.PlotShallowWaterErrorEvolution),
            ppr.DeleteArguments("problem", "benchmark"),
        ],
    }

    def __init__(self):
        self._parser = argparse.ArgumentParser(
            prog="cls1",
            description=textwrap.dedent(
                """\
            Explore different PDE-Solver for one-dimension conservation laws.

            """
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._add_command_parsers()

    def _add_command_parsers(self):
        parsers = self._parser.add_subparsers(
            title="Commands",
            dest="command",
            metavar="<command>",
            required=True,
        )

        self._add_help_parser(parsers)
        self._add_test_parser(parsers)
        self._add_calculation_parser(parsers)
        self._add_plot_parser(parsers)
        self._add_animate_parser(parsers)
        self._add_eoc_parser(parsers)
        self._add_plot_error_evolution_parser(parsers)

    def _add_test_parser(self, parsers):
        test_parser = parsers.add_parser(
            "test",
            help="Run unit test.",
            description="Task for running unit tests. If no argument is given run all tests.",
        )
        argument.add_file(test_parser)
        argument.add_profile(test_parser)
        argument.add_print_args(test_parser)

    def _add_help_parser(self, parsers):
        help_parser = parsers.add_parser(
            "help",
            help="Display help messages.",
            description="""Task for displaying help messages for solvers.
            Available solvers are: """
            + ", ".join([*SOLVER_PARSERS.keys()]),
        )
        argument.add_page(help_parser, SOLVER_PARSERS)
        argument.add_print_args(help_parser)

    def _add_calculation_parser(self, parsers):
        parser = parsers.add_parser(
            "calculate",
            help="Calculate solutions without doing with them something.",
            description="Calculate solutions without doing with them something.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        problem_parsers = self._add_problem_parsers(parser)

        for parser_func in [
            self._add_advection_parser,
            self._add_burgers_parser,
            self._add_shallow_water_parser,
        ]:
            problem_parser = parser_func(problem_parsers, "calculate")
            argument.add_profile(problem_parser)
            argument.add_print_args(problem_parser)

    def _add_problem_parsers(self, parser):
        return parser.add_subparsers(
            title="Problems", dest="problem", metavar="<problem>", required=True
        )

    def _add_advection_parser(self, parsers, command):
        parser = parsers.add_parser(
            "advection",
            help="Solve linear Advection.",
            description="""Solve liner Advection. For more information use 'cls1
            help SOLVER'.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        argument.add_benchmark(
            parser,
            advection.BENCHMARKS,
            advection.BENCHMARK_DEFAULTS[command],
        )
        argument.add_end_time(parser)
        argument.add_solver_argument(parser, action.ScalarSolverAction)

        return parser

    def _add_burgers_parser(self, parsers, command):
        parser = parsers.add_parser(
            "burgers",
            help="Solve Burgers.",
            description="""Solve Burgers. For more information use 'cls1
            help SOLVER'.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        argument.add_benchmark(
            parser,
            burgers.BENCHMARKS,
            burgers.BENCHMARK_DEFAULTS[command],
        )
        argument.add_end_time(parser)
        argument.add_solver_argument(parser, action.ScalarSolverAction)

        return parser

    def _add_shallow_water_parser(self, parsers, command):
        parser = parsers.add_parser(
            "swe",
            help="Solve Shallow Water equations.",
            description="""Solve shallow water equations (SWE). For more
            information use 'cls1 help SOLVER'.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        argument.add_benchmark(
            parser,
            shallow_water.BENCHMARKS,
            shallow_water.BENCHMARK_DEFAULTS[command],
        )
        argument.add_end_time(parser)
        argument.add_solver_argument(parser, action.ShallowWaterSolverAction)

        return parser

    def _add_plot_parser(self, parsers):
        parser = parsers.add_parser(
            "plot",
            help="Calculate and plot solutions.",
            description="Plot benchmarks and computed solutions.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        problem_parsers = self._add_problem_parsers(parser)

        for parser_func in [
            self._add_advection_parser,
            self._add_burgers_parser,
            self._add_shallow_water_parser,
        ]:
            problem_parser = parser_func(problem_parsers, "plot")
            argument.add_plot_mesh_size(problem_parser)
            argument.add_initial_data(problem_parser)
            argument.add_save(problem_parser, defaults.PLOT_TARGET)
            argument.add_hide(problem_parser)
            argument.add_profile(problem_parser)
            argument.add_print_args(problem_parser)

    def _add_animate_parser(self, parsers):
        parser = parsers.add_parser(
            "animate",
            help="Calculate and animate solutions.",
            description="Animate benchmarks and computed solutions.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        problem_parsers = self._add_problem_parsers(parser)

        for parser_func in [
            self._add_advection_parser,
            self._add_burgers_parser,
            self._add_shallow_water_parser,
        ]:
            problem_parser = parser_func(problem_parsers, "plot")
            argument.add_plot_mesh_size(problem_parser)
            argument.add_time_steps(problem_parser)
            argument.add_initial_data(problem_parser)
            argument.add_start_time(problem_parser)
            argument.add_save(problem_parser, defaults.ANIMATION_TARGET)
            argument.add_duration(problem_parser)
            argument.add_hide(problem_parser)
            argument.add_profile(problem_parser)
            argument.add_print_args(problem_parser)

    def _add_eoc_parser(self, parsers):
        parser = parsers.add_parser(
            "eoc",
            help="Compute Experimental convergence order (EOC).",
            description="Compute experimental order of convergence (EOC).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        problem_parsers = self._add_problem_parsers(parser)
        for parser_func in [self._add_advection_parser, self._add_burgers_parser]:
            problem_parser = parser_func(problem_parsers, "eoc")
            argument.add_eoc_mesh_size(problem_parser)
            argument.add_refine(problem_parser)
            argument.add_profile(problem_parser)
            argument.add_print_args(problem_parser)

    def _add_plot_error_evolution_parser(self, parsers):
        parser = parsers.add_parser(
            "plot-error-evolution",
            help="Plot error between two solutions.",
            description="Plot error between two solutions. Note, two solutions are required. The second one should be the reference solution.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        problem_parsers = self._add_problem_parsers(parser)
        for parser_func in [self._add_shallow_water_parser]:
            problem_parser = parser_func(problem_parsers, "plot-error-evolution")
            argument.add_save(problem_parser, defaults.ERROR_EVOLUTION_PATH)
            argument.add_hide(problem_parser)
            argument.add_profile(problem_parser)
            argument.add_print_args(problem_parser)

    def parse_arguments(self, *arguments) -> argparse.Namespace:
        arguments = self._parser.parse_args(*arguments)

        self._process_arguments(arguments)

        return arguments

    def _process_arguments(self, arguments):
        for process in self._arguments_postprocessing[arguments.command]:
            process(arguments)
