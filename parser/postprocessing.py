from typing import Type

import command as cmd
from benchmark.shallow_water import OscillationNoTopographyBenchmark
from tqdm.auto import tqdm, trange
from finite_volume.shallow_water.solver import ESTIMATOR_TYPES


################################################################################
# GENERAL POSTPROCESSING
################################################################################
class BuildCommand:
    command: Type[cmd.Command]

    def __init__(self, command: Type[cmd.Command]):
        self.command = command

    def __call__(self, arguments):
        arguments.command = self.command


class DeleteArguments:
    def __init__(self, *arguments):
        self.arguments = arguments

    def __call__(self, arguments):
        for argument in self.arguments:
            delattr(arguments, argument)


################################################################################
# BENCHMARK
################################################################################
def adjust_end_time(arguments):
    if arguments.end_time:
        arguments.benchmark.end_time = arguments.end_time

    del arguments.end_time


################################################################################
# SOLVER
################################################################################
def build_solver(arguments):
    solver_list = []

    for solver_arguments in arguments.solver:
        solver = solver_arguments.solver
        del solver_arguments.solver

        solver_list.append(solver(arguments.benchmark, **vars(solver_arguments)))

    arguments.solver = solver_list


################################################################################
# PLOT
################################################################################
def build_plotter(arguments):
    plotter = {
        "advection": cmd.ScalarPlotter,
        "burgers": cmd.ScalarPlotter,
        "swe": cmd.ShallowWaterPlotter,
    }

    arguments.plotter = plotter[arguments.problem](
        arguments.benchmark,
        mesh_size=arguments.mesh_size,
        save=arguments.save,
        show=arguments.show,
    )

    del arguments.mesh_size
    del arguments.save
    del arguments.show


################################################################################
# Animate
################################################################################
def build_animator(arguments):
    animator = {
        "advection": cmd.ScalarAnimator,
        "burgers": cmd.ScalarAnimator,
        "swe": cmd.ShallowWaterAnimator,
    }

    arguments.animator = animator[arguments.problem](
        arguments.benchmark,
        mesh_size=arguments.mesh_size,
        time_steps=arguments.time_steps,
        save=arguments.save,
        start_time=arguments.start_time,
        duration=arguments.duration,
        show=arguments.show,
    )

    del arguments.mesh_size
    del arguments.time_steps
    del arguments.save
    del arguments.start_time
    del arguments.duration
    del arguments.show


def add_save_history_argument(arguments):
    for solver_arguments in arguments.solver:
        solver_arguments.save_history = True


################################################################################
# EOC
################################################################################
def build_eoc_solutions(arguments):
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
            cmd.Calculate(solver, leave=False).execute()
            solvers.append(solver)
            solver_spaces.append(solver.solution.space)

        arguments.solvers.append(solvers)
        arguments.solver_spaces.append(solver_spaces)

    del arguments.refine
    del arguments.mesh_size
    del arguments.solver


################################################################################
# GENERATE DATA
################################################################################
def build_directory(arguments):
    directories = {"MCL": "data/reduced-mcl/", "LLF": "data/reduced-llf"}
    if arguments.directory is None:
        arguments.directory = directories[arguments.solver[0].short]


def extract_solver(arguments):
    solver_num = 0 if arguments.solver is None else len(arguments.solver)
    assert solver_num == 1, f"Exactly one solver must be given. There are {solver_num}."

    arguments.solver = arguments.solver[0]


def build_benchmark(arguments):
    arguments.benchmark = OscillationNoTopographyBenchmark()


def build_overwrite_argument(arguments):
    arguments.overwrite = not arguments.append
    del arguments.append


################################################################################
# TRAIN NETWORK
################################################################################
def build_train_network_arguments(arguments):
    arguments.estimator_type = ESTIMATOR_TYPES[arguments.network]

    directories = {"llf": "data/reduced-llf/", "mcl": "data/reduced-mcl/"}
    arguments.data_path = directories[arguments.network] + "data.csv"
    arguments.network_path = directories[arguments.network] + arguments.file + ".pkl"

    del arguments.network
    del arguments.file
