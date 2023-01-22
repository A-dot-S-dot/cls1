from typing import Sequence

from core import Solver
from tqdm.auto import tqdm

from .command import Command


class Calculate(Command):
    """Calculate discrete solution without doing with it something."""

    _solver: Solver | Sequence[Solver]
    _leave_solution_progressbar: bool

    def __init__(
        self, solver: Solver | Sequence[Solver], leave_solution_progress_bar=True
    ):
        self._solver = solver
        self._leave_solution_progressbar = leave_solution_progress_bar

    def execute(self):
        if isinstance(self._solver, Solver):
            self._solver.solve(leave_progress_bar=self._leave_solution_progressbar)
        elif len(self._solver) == 1:
            self._solver[0].solve(leave_progress_bar=self._leave_solution_progressbar)
        else:
            for solver in tqdm(
                self._solver, desc="Calculate", unit="solver", leave=False
            ):
                solver.solve(leave_progress_bar=self._leave_solution_progressbar)
