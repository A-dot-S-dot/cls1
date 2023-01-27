from typing import Dict, Sequence

from core import Solver
from tqdm.auto import tqdm

from .command import Command


class Calculate(Command):
    """Calculate discrete solution without doing with it something."""

    _solver: Solver | Sequence[Solver]
    _tqdm_kwargs: Dict

    def __init__(self, solver: Solver | Sequence[Solver], **tqdm_kwargs):
        self._solver = solver
        self._tqdm_kwargs = tqdm_kwargs

    def execute(self):
        if isinstance(self._solver, Solver):
            self._solver.solve(**self._tqdm_kwargs)
        elif len(self._solver) == 1:
            self._solver[0].solve(**self._tqdm_kwargs)
        else:
            for solver in tqdm(
                self._solver, desc="Calculate", unit="solver", leave=False
            ):
                solver.solve(**self._tqdm_kwargs)
