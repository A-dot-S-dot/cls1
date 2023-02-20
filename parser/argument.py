import defaults
from core import ode_solver
from shallow_water import RandomOscillationNoTopographyBenchmark
from shallow_water.solver import SHALLOW_WATER_FLUX_GETTER

from . import parser_type


################################################################################
# GENERAL
################################################################################
def add_profile(parser):
    parser.add_argument(
        "--profile",
        help="Profile program for optimization purposes. You can specify the number of lines to be printed. Otherwise print %(const)s lines.",
        type=int,
        nargs="?",
        const=30,
        default=0,
        metavar="<lines>",
    )


def add_print_args(parser):
    parser.add_argument("--args", action="store_true", help="Print given arguments.")


################################################################################
# TEST
################################################################################
def add_file(parser):
    parser.add_argument(
        "file",
        nargs="*",
        help="Run unittest contained in FILE_test.py.",
        metavar="<file>",
    )


################################################################################
# HELP
################################################################################
def add_page(parser, solver_parsers):
    parser.add_argument(
        "parser",
        help="Specify page which should be displayed in terminal.",
        type=lambda input: solver_parsers[input],
        metavar="<page>",
    )


################################################################################
# COMMANDS
################################################################################
def add_benchmark(parser, benchmarks, default_benchmark):
    parser.add_argument(
        "-b",
        "--benchmark",
        help="""Choose benchmark by key. Available keys are: """
        + ", ".join([*benchmarks.keys()]),
        type=lambda input: benchmarks[input](),
        metavar="<name>",
        default=default_benchmark(),
    )


def add_random_shallow_water_benchmark(parser):
    parser.add_argument(
        "-r",
        "--random-benchmark",
        help="Choose random benchmark.",
        type=lambda input: RandomOscillationNoTopographyBenchmark(seed=int(input)),
        nargs="?",
        const=RandomOscillationNoTopographyBenchmark(),
        metavar="<seed>",
        dest="benchmark",
    )


def add_end_time(parser):
    parser.add_argument(
        "-T",
        "--end-time",
        type=parser_type.positive_float,
        help="End time used by solvers. If not specified use benchmark's end time.",
        metavar="<time>",
    )


def add_solver(parser, action):
    parser.add_argument(
        "-s",
        "--solver",
        help="Solver on which the task is applied.",
        nargs="+",
        action=action,
        metavar="<solver> [options]",
    )


def add_plot_mesh_size(parser):
    parser.add_argument(
        "-m",
        "--mesh-size",
        help="Number of points used for plotting.",
        type=parser_type.positive_int,
        metavar="<size>",
        default=defaults.PLOT_MESH_SIZE,
    )


def add_initial_data(parser):
    parser.add_argument("--initial", help="Show initial data.", action="store_true")


def add_save(parser, target: str):
    parser.add_argument(
        "--save",
        help=f"Save file in specified direction. (const: {target})",
        nargs="?",
        const=target,
        metavar="<file>",
    )


def add_hide(parser):
    parser.add_argument(
        "--hide", help=f"Do not show any figures.", action="store_false", dest="show"
    )


def add_time_steps(parser):
    parser.add_argument(
        "--time_steps",
        help="Specify how many time steps should be at least simulated.",
        type=parser_type.positive_int,
        metavar="<number>",
        default=defaults.TIME_STEPS,
    )


def add_start_time(parser):
    parser.add_argument(
        "-t",
        "--start-time",
        help="Set start time for animation.",
        type=parser_type.positive_float,
        metavar="<time>",
    )


def add_duration(parser):
    parser.add_argument(
        "--duration",
        help="""Specifies how many second an animation should last.""",
        type=parser_type.positive_float,
        default=defaults.DURATION,
        metavar="<factor>",
    )


def add_eoc_mesh_size(parser):
    parser.add_argument(
        "-m",
        "--mesh-size",
        help="""Initial mesh size for eoc calculation. Note, that mesh size
        arguments for solvers are not considered.""",
        type=parser_type.positive_int,
        metavar="<size>",
        default=defaults.EOC_MESH_SIZE,
    )


def add_refine(parser):
    parser.add_argument(
        "-r",
        "--refine",
        help="Specify number of refinements.",
        type=parser_type.positive_int,
        default=defaults.REFINE_NUMBER,
        metavar="<number>",
    )


################################################################################
# SOLVER
################################################################################
def add_name(parser, default):
    parser.add_argument(
        "+n",
        "++name",
        type=str,
        help="Specify short name",
        metavar="<name>",
        default=default,
    )


def add_short(parser, default):
    parser.add_argument(
        "+s",
        "++short",
        type=str,
        help="Specify short name",
        metavar="<short>",
        default=default,
    )


def add_mesh_size(parser, default=None):
    parser.add_argument(
        "+m",
        "++mesh-size",
        help="Number of mesh cells. If not specified use the default size for the chosen task.",
        type=parser_type.positive_int,
        metavar="<size>",
        default=default or defaults.CALCULATE_MESH_SIZE,
    )


def add_cfl_number(parser, default: float):
    parser.add_argument(
        "++cfl",
        help="Specify the cfl number for time stepping.",
        type=parser_type.positive_float,
        metavar="<number>",
        dest="cfl_number",
        default=default,
    )


def add_adaptive_time_stepping(parser):
    parser.add_argument(
        "++adaptive",
        help="Make time stepping adaptive, if available.",
        action="store_true",
    )


# FINITE ELEMENT
def add_polynomial_degree(parser):
    parser.add_argument(
        "+p",
        "++polynomial-degree",
        help="Set polynomial degree used for finite elements.",
        metavar="<degree>",
        type=parser_type.positive_int,
        default=defaults.POLYNOMIAL_DEGREE,
    )


def add_exact_flux(parser):
    parser.add_argument(
        "++exact-flux", action="store_true", help="Calculate flux matrices exactly."
    )


def add_ode_solver(parser):
    _ode_solver = {
        "euler": ode_solver.ForwardEuler,
        "heun": ode_solver.Heun,
        "ssp3": ode_solver.StrongStabilityPreservingRungeKutta3,
        # "ssp4": ode_solver.StrongStabilityPreservingRungeKutta4,
    }

    parser.add_argument(
        "++ode-solver",
        help="Specify ode solver.",
        type=lambda input: _ode_solver[input],
        metavar="<solver>",
        dest="ode_solver_type",
        default=_ode_solver[defaults.ODE_SOLVER],
    )


# COARSE SOLVER
def add_coarsening_degree(parser):
    parser.add_argument(
        "+c",
        "++coarsening-degree",
        help="Specify the coarsening degree.",
        type=parser_type.positive_int,
        metavar="<degree>",
        default=defaults.COARSENING_DEGREE,
    )


def add_network_load_path(parser):
    parser.add_argument(
        "++network-path",
        help="Specify from where to load trained network.",
        metavar="<file>",
        default=defaults.NETWORK_PATH,
    )


# LIMITER
def add_flux_getter(parser):
    parser.add_argument(
        "+f",
        "++flux",
        help="""Choose flux by key. Available keys are: """
        + ", ".join([*SHALLOW_WATER_FLUX_GETTER.keys()]),
        type=lambda input: SHALLOW_WATER_FLUX_GETTER[input],
        metavar="<flux>",
        dest="flux_getter",
    )


# ANTIDIFFUSION
def add_antidiffusion_gamma(parser):
    parser.add_argument(
        "+g",
        "++gamma",
        help="Specify antidiffusion parameter",
        type=float,
        metavar="<gamma>",
        default=defaults.ANTIDIFFUSION_GAMMA,
    )
