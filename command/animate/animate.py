import time
from argparse import Namespace

import numpy as np
from benchmark import Benchmark, NoExactSolutionError, SWEBenchmark
from command import Command
from defaults import PLOT_MESH_SIZE
from factory.pde_solver_factory import PDESolverFactory
from pde_solver.discrete_solution import DiscreteSolution
from pde_solver.solver_components import SolverComponents
from tqdm import tqdm

from .animator import NothingToPlotError, SolutionAnimator
from .scalar import ScalarFunctionAnimator
from .swe import SWEFunctionAnimator


class AnimateCommand(Command):
    _args: Namespace
    _benchmark: Benchmark
    _components: SolverComponents
    _animator: SolutionAnimator

    def __init__(self, args: Namespace):
        self._args = args
        self._components = SolverComponents(args)
        self._benchmark = self._components.benchmark

        self._build_animator()
        self._build_grid()

    def _build_animator(self):
        if isinstance(self._benchmark, SWEBenchmark):
            self._animator = SWEFunctionAnimator(self._benchmark)
        else:
            self._animator = ScalarFunctionAnimator(self._benchmark)

        self._animator.interval = self._args.animate.interval

    def _build_grid(self):
        self._animator.set_grid(self._benchmark.domain, PLOT_MESH_SIZE)

    def execute(self):
        self._add_animations()
        self._animator.set_suptitle(f"{len(self._components.mesh)} cells")

        try:
            self._animator.show()
        except NothingToPlotError:
            tqdm.write("WARNING: Nothing to plot...")

    def _add_animations(self):
        self._add_discrete_solutions()
        self._add_exact_solution()

        if self._args.animate.initial:
            self._animator.add_initial_data()

    def _add_exact_solution(self):
        if len(self._animator.temporal_grid) == 0:
            self._animator.temporal_grid = np.linspace(
                self._benchmark.start_time, self._benchmark.end_time, 400
            )

        try:
            self._animator.add_exact_solution()
        except NoExactSolutionError as error:
            tqdm.write("WARNING: " + str(error))

    def _add_discrete_solutions(self):
        for solver_factory in tqdm(
            self._components.solver_factories,
            desc="Calculate solutions",
            unit="solver",
            leave=False,
        ):
            self._add_discrete_solution(solver_factory)

    def _add_discrete_solution(self, solver_factory: PDESolverFactory):
        solution = self._solve_pde(solver_factory)
        self._animate_discrete_solution(solution, solver_factory)

    def _solve_pde(self, solver_factory: PDESolverFactory) -> DiscreteSolution:
        solver = solver_factory.solver

        start_time = time.time()
        solver.solve()

        tqdm.write(
            f"Solved {solver_factory.info} with {solver_factory.dimension} DOFs and {solver.time_stepping.time_steps} time steps in {time.time()-start_time:.2f}s."
        )

        return solver.solution

    def _animate_discrete_solution(
        self, solution: DiscreteSolution, solver_factory: PDESolverFactory
    ):
        self._animator.temporal_grid = np.array(solution.time)
        self._animator.add_function_values(
            solver_factory.grid,
            np.array(solution.time),
            solution.values,
            *solver_factory.plot_label,
        )
