import argparse
import textwrap

import command
import defaults
from benchmark import advection, burgers, shallow_water
from finite_volume.shallow_water import command as shallow_water_command

from . import action, argument
from . import postprocessing as ppr
from .solver_parser import SOLVER_PARSERS


class CustomArgumentParser:
    """Parser for command line arguments."""

    _arguments_postprocessing = {
        "test": [ppr.BuildCommand(command.Test)],
        "help": [ppr.BuildCommand(command.Help)],
        "calculate": [
            ppr.adjust_end_time,
            ppr.build_solver,
            ppr.BuildCommand(command.Calculate),
            ppr.DeleteArguments("problem", "benchmark"),
        ],
        "plot": [
            ppr.adjust_end_time,
            ppr.build_solver,
            ppr.build_plotter,
            ppr.BuildCommand(command.Plot),
            ppr.DeleteArguments("problem"),
        ],
        "animate": [
            ppr.adjust_end_time,
            ppr.add_save_history_argument,
            ppr.build_solver,
            ppr.build_animator,
            ppr.BuildCommand(command.Animate),
            ppr.DeleteArguments("problem"),
        ],
        "eoc": [
            ppr.adjust_end_time,
            ppr.build_eoc_solutions,
            ppr.BuildCommand(command.CalculateEOC),
            ppr.DeleteArguments("problem"),
        ],
        "generate-data": [
            ppr.build_directory,
            ppr.build_overwrite_argument,
            ppr.build_benchmark,
            ppr.add_save_history_argument,
            ppr.build_solver,
            ppr.extract_solver,
            ppr.BuildCommand(shallow_water_command.GenerateData),
            ppr.DeleteArguments("benchmark"),
        ],
        "analyze-data": [ppr.BuildCommand(command.AnalyzeData)],
        "analyze-curvature": [
            ppr.BuildCommand(shallow_water_command.PlotCurvatureAgainstSubgridFlux)
        ],
        "train-network": [
            ppr.build_train_network_arguments,
            ppr.BuildCommand(shallow_water_command.TrainNetwork),
        ],
        "plot-error-evolution": [
            ppr.adjust_end_time,
            ppr.add_save_history_argument,
            ppr.build_solver,
            ppr.BuildCommand(shallow_water_command.PlotShallowWaterErrorEvolution),
            ppr.DeleteArguments("benchmark"),
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
        self._add_generate_data_parser(parsers)
        self._add_analyze_data_parser(parsers)
        self._add_analyze_curvature_parser(parsers)
        self._add_train_network_parser(parsers)
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
        argument.add_solver(parser, action.ScalarSolverAction)

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
        argument.add_solver(parser, action.ScalarSolverAction)

        return parser

    def _add_shallow_water_parser(self, parsers, command):
        parser = parsers.add_parser(
            "swe",
            help="Solve Shallow Water equations.",
            description="""Solve shallow water equations (SWE). For more
            information use 'cls1 help SOLVER'.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_shallow_water_benchmark_group(parser, command)
        argument.add_end_time(parser)
        argument.add_solver(parser, action.ShallowWaterSolverAction)

        return parser

    def _add_shallow_water_benchmark_group(self, parser, command):
        benchmark_group = parser.add_mutually_exclusive_group()
        argument.add_benchmark(
            benchmark_group,
            shallow_water.BENCHMARKS,
            shallow_water.BENCHMARK_DEFAULTS[command],
        )
        argument.add_random_shallow_water_benchmark(benchmark_group)

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

    def _add_generate_data_parser(self, parsers):
        parser = parsers.add_parser(
            "generate-data",
            help="Generate data for reduced shallow water models.",
            description="""Generates data for shallow water models which uses
            neural networks subgrid fluxes.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        argument.add_directory(parser)
        argument.add_end_time(parser)
        argument.add_solver(parser, action.ShallowWaterSolverAction)
        argument.add_solution_number(parser)
        argument.add_seed(parser)
        argument.add_input_radius(parser)
        argument.add_node_index(parser)
        argument.add_append(parser)
        argument.add_profile(parser)
        argument.add_print_args(parser)

    def _add_analyze_data_parser(self, parsers):
        parser = parsers.add_parser(
            "analyze-data",
            help="Analyze data.",
            description="""Analye data by printing summary statistics and historgrams.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        argument.add_data_path(parser)
        argument.add_histogram(parser)
        argument.add_profile(parser)
        argument.add_print_args(parser)

    def _add_analyze_curvature_parser(self, parsers):
        parser = parsers.add_parser(
            "analyze-curvature",
            help="Analyze curvature.",
            description="""Analye curvature by plotting it against subgrid flux.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        argument.add_hide(parser)
        argument.add_profile(parser)
        argument.add_print_args(parser)

    def _add_train_network_parser(self, parsers):
        parser = parsers.add_parser(
            "train-network",
            help="Trains networks for reduced models.",
            description="""Train networks for reduced shallow water models.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        argument.add_network(parser)
        argument.add_network_file_name(parser)
        argument.add_epochs(parser)
        argument.add_skip(parser)
        argument.add_plot_loss(parser)
        argument.add_profile(parser)
        argument.add_print_args(parser)

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

        argument.add_end_time(parser)
        self._add_shallow_water_benchmark_group(parser, "calculate")
        argument.add_solver(parser, action.ShallowWaterSolverAction)
        argument.add_save(parser, defaults.ERROR_EVOLUTION_PATH)
        argument.add_hide(parser)
        argument.add_profile(parser)
        argument.add_print_args(parser)

    def parse_arguments(self, *arguments) -> argparse.Namespace:
        arguments = self._parser.parse_args(*arguments)

        self._process_arguments(arguments)

        return arguments

    def _process_arguments(self, arguments):
        for process in self._arguments_postprocessing[arguments.command]:
            process(arguments)
