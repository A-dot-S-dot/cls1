import argparse
import textwrap

import custom_type
from defaults import *

from .action import *
from .solver_parser import SOLVER_PARSERS

AVAILABLE_HELP_ARGUMENTS = ", ".join(
    [*SOLVER_PARSERS.keys(), "benchmark", "plot", "animate", "eoc", "calculate"]
)


class CustomArgumentParser:
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

    def __init__(self):
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
        self._add_advection_parser(parsers)
        self._add_burgers_parser(parsers)
        self._add_shallow_water_parser(parsers)
        self._add_generate_data_parser(parsers)
        self._add_train_network_parser(parsers)

    def _add_test_parser(self, parsers):
        test_parser = parsers.add_parser(
            "test",
            help="Run unit test.",
            description="Task for running unit tests. If no argument is given run all tests.",
        )
        self._add_file(test_parser)
        self._add_profile(test_parser)
        self._add_args(test_parser)

    def _add_file(self, parser):
        parser.add_argument(
            "test-file",
            nargs="*",
            help="Run unittest contained in FILE_test.py.",
            metavar="<file>",
        )

    def _add_profile(self, parser):
        parser.add_argument(
            "--profile",
            help="Profile program for optimization purposes. You can specify the number of lines to be printed. Otherwise print %(const)s lines.",
            type=int,
            nargs="?",
            const=30,
            default=0,
            metavar="<lines>",
        )

    def _add_args(self, parser):
        parser.add_argument(
            "--args", action="store_true", help="Print given arguments."
        )

    def _add_help_parser(self, parsers):
        help_parser = parsers.add_parser(
            "help",
            help="Display help messages.",
            description="Task for displaying different help messages for certain objects. Available arguments are: "
            + AVAILABLE_HELP_ARGUMENTS,
        )
        self._add_page(help_parser)
        self._add_args(help_parser)

    def _add_page(self, parser):
        parser.add_argument(
            "page",
            help="Specify page which should be displayed in terminal.",
            metavar="<page>",
        )
        parser.add_argument(
            "option",
            help="Additional option for page if it is available. ",
            nargs="*",
            metavar="<option>",
        )

    def _add_advection_parser(self, parsers):
        advection_parser = parsers.add_parser(
            "advection",
            help="Solve linear Advection.",
            description="""Solver liner Advection. Available solvers are 'cg',
            'low_cg' and 'mcl'. For more information use 'help' program.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_command(advection_parser)
        self._add_benchmark(advection_parser)
        self._add_solver(advection_parser, AdvectionSolverAction)
        self._add_mesh_size(advection_parser)
        self._add_end_time(advection_parser)
        self._add_profile(advection_parser)
        self._add_args(advection_parser)

    def _add_command(
        self,
        parser,
        plot=True,
        animate=True,
        eoc=True,
        calculation=True,
    ):
        task_group = parser.add_mutually_exclusive_group(required=True)

        if plot:
            self._add_plot(task_group)
        if animate:
            self._add_animate(task_group)
        if eoc:
            self._add_eoc(task_group)
        if calculation:
            self._add_calculate(task_group)

    def _add_plot(self, parser):
        parser.add_argument(
            "-p",
            "--plot",
            help="Plot discrete solution. For more information use 'help' program.",
            nargs="*",
            action=PlotAction,
            metavar="<arguments>",
        )

    def _add_animate(self, parser):
        parser.add_argument(
            "-a",
            "--animate",
            help="Animate discrete solution. For more information use 'help' program.",
            nargs="*",
            action=AnimateAction,
            metavar="<arguments>",
        )

    def _add_eoc(self, parser):
        parser.add_argument(
            "-e",
            "--eoc",
            help="Compute eoc. For more information use 'help' program.",
            nargs="*",
            action=EOCAction,
            metavar="<arguments>",
        )

    def _add_calculate(self, parser):
        parser.add_argument(
            "-c",
            "--calculate",
            help="Calculate solutions without doing with them something.",
            nargs="*",
            action=CalculationAction,
            metavar="<arguments>",
        )

    def _add_mesh_size(self, parser):
        parser.add_argument(
            "-m",
            "--mesh-size",
            help="Number of mesh cells. If not specified use the default size for the chosen task.",
            type=custom_type.positive_int,
            metavar="<size>",
        )

    def _add_benchmark(self, parser):
        parser.add_argument(
            "-b",
            "--benchmark",
            help="""Benchmark for conservation law.""",
            type=custom_type.non_negative_int,
            metavar="<benchmark>",
        )

    def _add_end_time(self, parser):
        parser.add_argument(
            "-T",
            "--end-time",
            type=custom_type.positive_float,
            help="End time used by solvers. If not specified use benchmark's end time.",
            metavar="<time>",
        )

    def _add_solver(self, parser, action):
        parser.add_argument(
            "-s",
            "--solver",
            help="Solver on which the task is applied.",
            nargs="+",
            action=action,
            metavar="<solver> [options]",
        )

    def _add_burgers_parser(self, parsers):
        burgers_parser = parsers.add_parser(
            "burgers",
            help="Solve Burgers.",
            description="""Solve Burgers. Available solvers are 'cg',
            'low_cg' and 'mcl'. For more information use 'help' program.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_command(burgers_parser)
        self._add_solver(burgers_parser, BurgersSolverAction)
        self._add_benchmark(burgers_parser)
        self._add_mesh_size(burgers_parser)
        self._add_end_time(burgers_parser)
        self._add_profile(burgers_parser)
        self._add_args(burgers_parser)

    def _add_shallow_water_parser(self, parsers):
        shallow_water_parser = parsers.add_parser(
            "swe",
            help="Solve Shallow Water equations.",
            description="""Solve shallow water equations (SWE). For more
            information use 'help' program.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_command(
            shallow_water_parser,
            eoc=False,
        )
        self._add_benchmark(shallow_water_parser)
        self._add_solver(shallow_water_parser, SWESolverAction)
        self._add_mesh_size(shallow_water_parser)
        self._add_end_time(shallow_water_parser)
        self._add_profile(shallow_water_parser)
        self._add_args(shallow_water_parser)

    def _add_generate_data_parser(self, parsers):
        parser = parsers.add_parser(
            "generate-data",
            help="Generate data for network training.",
            description="""Generate data for network training. The network is
            used to predict subgrid fluxes for Shallow Water.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_solver_mesh_size(parser)
        self._add_coarsening_degree(parser)
        self._add_cfl_number(parser)
        self._add_local_degree(parser)
        self._add_solutions_number(parser)
        self._add_skip_time_steps(parser)
        self._add_training_data_save_path(parser)
        self._add_validation_data_save_path(parser)
        self._add_benchmark_parameters_save_path(parser)
        self._add_append(parser)
        self._add_profile(parser)
        self._add_args(parser)

    def _add_solver_mesh_size(self, parser):
        parser.add_argument(
            "-m",
            "--mesh-size",
            help="Number of mesh cells.",
            type=custom_type.positive_int,
            metavar="<size>",
            default=CALCULATION_MESH_SIZE,
        )

    def _add_coarsening_degree(self, parser):
        parser.add_argument(
            "--coarsening-degree",
            help="Specify the coarsening degree.",
            type=custom_type.positive_int,
            metavar="<degree>",
            default=COARSENING_DEGREE,
        )

    def _add_cfl_number(self, parser):
        parser.add_argument(
            "--cfl",
            help="Set CFL number for time stepping.",
            type=custom_type.positive_float,
            metavar="<number>",
            default=GODUNOV_CFL_NUMBER,
            dest="cfl_number",
        )

    def _add_local_degree(self, parser):
        parser.add_argument(
            "--local-degree",
            help="""Specify how many neighboured cell values of an edge should be considered.""",
            type=custom_type.positive_int,
            metavar="<degree>",
            default=LOCAL_DEGREE,
        )

    def _add_solutions_number(self, parser):
        parser.add_argument(
            "--solution-number",
            help="Specify how many solutions for the data set should be calculated.",
            type=custom_type.positive_int,
            metavar="<number>",
            default=SOLUTION_NUMBER,
        )

    def _add_skip_time_steps(self, parser):
        parser.add_argument(
            "--skip-steps",
            help="Specify how many time steps should be skipped, i.e. save each SKIP-STEPS-th time step.",
            type=custom_type.positive_int,
            metavar="<steps>",
            default=SKIP_STEPS,
        )

    def _add_training_data_save_path(self, parser):
        parser.add_argument(
            "--train-path",
            help="Specify where to save training data.",
            metavar="<file>",
            default=TRAINING_DATA_PATH,
        )

    def _add_validation_data_save_path(self, parser):
        parser.add_argument(
            "--validate-path",
            help="Specify where to save validation data.",
            metavar="<file>",
            default=VALIDATION_DATA_PATH,
        )

    def _add_benchmark_parameters_save_path(self, parser):
        parser.add_argument(
            "--benchmark-parameters-path",
            help="Specify where to save benchmark parameters.",
            metavar="<file>",
            default=BENCHMARK_PARAMETERS_PATH,
        )

    def _add_append(self, parser):
        parser.add_argument(
            "--append",
            help="""After generating data append it in the save path if
            there is already some data.""",
            action="store_true",
        )

    def _add_train_network_parser(self, parsers):
        parser = parsers.add_parser(
            "train-network",
            help="Train network for predicting subfluxes.",
            description="""Train network which is used to predict subgrid fluxes
            for Shallow Water""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_epochs(parser)
        self._add_batch_size(parser)
        self._add_hidden_layers(parser)
        self._add_learning_rate_group(parser)
        self._add_suptitle(parser)
        self._add_network_save_path(parser)
        self._add_training_data_load_path(parser)
        self._add_validation_data_load_path(parser)
        self._add_profile(parser)
        self._add_args(parser)

    def _add_epochs(self, parser):
        parser.add_argument(
            "-e",
            "--epochs",
            help="Set how many training iterations should be performend.",
            type=custom_type.positive_int,
            metavar="<epochs>",
            default=EPOCHS,
        )

    def _add_batch_size(self, parser):
        parser.add_argument(
            "-b",
            "--batch-size",
            help="Set batch size.",
            type=custom_type.positive_int,
            metavar="<size>",
            default=BATCH_SIZE,
        )

    def _add_hidden_layers(self, parser):
        parser.add_argument(
            "-l",
            "--hidden-neurons",
            help="Specify hidden neurons per hidden layer.",
            type=custom_type.positive_int,
            nargs="*",
            metavar="<neurons>",
            default=HIDDEN_NEURONS,
        )

    def _add_learning_rate_group(self, parser):
        group = parser.add_argument_group("Learning Rate")
        self._add_learning_rate(group)
        self._add_learning_rate_patience(group)
        self._add_learning_rate_factor(group)

    def _add_learning_rate(self, parser):
        parser.add_argument(
            "--learning-rate",
            help="Set learning rate.",
            type=custom_type.positive_float,
            metavar="<rate>",
            default=LEARNING_RATE,
        )

    def _add_learning_rate_patience(self, parser):
        parser.add_argument(
            "--patience",
            help="Set how long learning rate scheduler should at least wait for an update after decreasing learning rate.",
            type=custom_type.positive_int,
            metavar="<patience>",
            default=LEARNING_RATE_UPDATE_PATIENCE,
        )

    def _add_learning_rate_factor(self, parser):
        parser.add_argument(
            "--factor",
            help="Set decreasing factor for learning rate.",
            type=custom_type.percent_number,
            metavar="<factor>",
            default=LEARNING_RATE_DECREASING_FACTOR,
        )

    def _add_suptitle(self, parser):
        parser.add_argument(
            "--suptitle",
            help="Ser suptitle for loss plot.",
            metavar="<title>",
            default="Losses",
        )

    def _add_network_save_path(self, parser):
        parser.add_argument(
            "--network-path",
            help="Specify where to save trained network.",
            metavar="<file>",
            default=NETWORK_PATH,
        )

    def _add_training_data_load_path(self, parser):
        parser.add_argument(
            "--train-path",
            help="Specify from where to load training data.",
            metavar="<file>",
            default=TRAINING_DATA_PATH,
        )

    def _add_validation_data_load_path(self, parser):
        parser.add_argument(
            "--validate-path",
            help="Specify from where to load validation data.",
            metavar="<file>",
            default=VALIDATION_DATA_PATH,
        )

    def parse_arguments(self, *args) -> argparse.Namespace:
        return self._parser.parse_args(*args)
