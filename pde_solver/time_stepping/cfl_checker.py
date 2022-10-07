import numpy as np
from pde_solver.system_vector import SystemVector
from tqdm import tqdm
from .time_stepping import TimeStepping


class CFLCheckedVector(SystemVector):
    """Decorates a system vector such that cfl condition is checked every time
    it is invoked.

    """

    _component: SystemVector
    _time_stepping: TimeStepping

    _last_dof_vector: np.ndarray
    _last_value: np.ndarray

    def __init__(self, component: SystemVector, time_stepping: TimeStepping):
        self._component = component
        self._time_stepping = time_stepping

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        value = self._component.__call__(dof_vector)

        if not self._time_stepping.satisfy_cfl():
            tqdm.write(
                f"WARNING: CFL condition is violated at time {self._time_stepping.time:.4f}"
            )

        return value
