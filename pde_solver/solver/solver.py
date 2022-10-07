from abc import ABC, abstractmethod
from typing import Dict

from pde_solver.discrete_solution import DiscreteSolutionObservable
from pde_solver.time_stepping import TimeStepping, TimeStepTooSmallError
from tqdm import tqdm


class PDESolver(ABC):
    solution: DiscreteSolutionObservable
    tqdm_kwargs: Dict
    time_stepping: TimeStepping

    def solve(self):
        progress_iterator = tqdm(self.time_stepping, **self.tqdm_kwargs)

        for _ in progress_iterator:
            try:
                self.update()
            except TimeStepTooSmallError:
                tqdm.write("WARNING: time step is too small calculation is interrupted")
                break

    @abstractmethod
    def update(self):
        ...
