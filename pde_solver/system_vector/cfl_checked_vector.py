import numpy as np
from pde_solver.time_stepping import TimeStepping
from tqdm import tqdm

from .system_vector import SystemVector


class CFLCheckedVector(SystemVector):
    """In some cases time stepping depends on a right hand side. With this
    decorator the cfl condition is checked for each time the right hand side is
    changed.

    """

    _vector: SystemVector
    _time_stepping: TimeStepping

    def __init__(self, vector: SystemVector, time_stepping: TimeStepping):
        self._vector = vector
        self._time_stepping = time_stepping

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        value = self._vector.__call__(dof_vector)

        if not self._time_stepping.satisfy_cfl_condition():
            tqdm.write(
                f"WARNING: CFL condition is violated at time {self._time_stepping.time:.4f}"
            )

        return value

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({self._vector})"
