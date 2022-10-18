from pde_solver.discrete_solution import (
    DiscreteSolution,
    DiscreteSolutionObservable,
    DiscreteSolutionObserver,
)
from pde_solver.system_matrix import SystemMatrix
from pde_solver.system_vector import SystemVector

from .time_stepping import TimeStepping


class MCLTimeStepping(TimeStepping):
    """This time stepping ensures that the time step for linear advection solvers is

        dt = CFL_NUMBER*min(mi/dii)

    where mi denotes the lumped mass and dij a diffusion. At the end dt could be
    smaller to reach end time.

    """

    lumped_mass: SystemVector
    artificial_diffusion: SystemMatrix
    discrete_solution: DiscreteSolution

    _default_time_step: float
    _update_needed = True

    @property
    def desired_time_step(self) -> float:
        return self.cfl_number * self._default_time_step

    def update_time_step(self):
        self._default_time_step = self.optimal_time_step

    @property
    def optimal_time_step(self) -> float:
        return min(
            self.lumped_mass() / (2 * abs(self.artificial_diffusion().diagonal()))
        )

    def satisfy_cfl(self) -> bool:
        return self.desired_time_step < self.optimal_time_step + 1e-8


class AdaptiveMCLTimeStepping(MCLTimeStepping, DiscreteSolutionObserver):
    """This time stepping ensures for each time step

        dt = CFL_NUMBER*min(mi/dii)

    where mi denotes the lumped mass and dij a diffusion. At the end dt could be
    smaller to reach end time.

    """

    def __init__(self, discrete_solution: DiscreteSolutionObservable):
        discrete_solution.register_observer(self)

    def update(self):
        self.update_time_step()
