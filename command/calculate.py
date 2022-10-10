from argparse import Namespace
import time

from tqdm import tqdm
from pde_solver.solver_components import SolverComponents

from .command import Command


class CalculationCommand(Command):
    """Calculate discrete solution without doing with it something."""

    def __init__(self, args: Namespace):
        self._args = args
        self._components = SolverComponents(args)

    def execute(self):
        for solver_factory in tqdm(
            self._components.solver_factories,
            desc="Calculate solutions",
            unit="solver",
            leave=False,
        ):
            solver = solver_factory.solver

            start_time = time.time()
            solver.solve()

            tqdm.write(
                f"Solved {solver_factory.info} with {solver_factory.dimension} DOFs and {solver.time_stepping.time_steps} time steps in {time.time()-start_time:.2f}s."
            )
