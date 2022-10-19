import numpy as np
from pde_solver.mesh import Mesh
from pde_solver.system_flux.swe_intermediate_velocities import SWEIntermediateVelocities

from .time_stepping import TimeStepping


class SWEGodunovTimeStepping(TimeStepping):
    mesh: Mesh
    intermediate_velocities: SWEIntermediateVelocities

    @property
    def desired_time_step(self) -> float:
        return self.cfl_number * self.optimal_time_step

    @property
    def optimal_time_step(self) -> float:
        return self.mesh.step_length / (
            2
            * np.max(
                [
                    abs(self.intermediate_velocities.left_velocities),
                    self.intermediate_velocities.right_velocities,
                ]
            )
        )


class SWEGodunovConstantTimeStepping(SWEGodunovTimeStepping):
    _default_time_step: float

    @property
    def desired_time_step(self) -> float:
        return self.cfl_number * self._default_time_step

    def update_time_step(self):
        self._default_time_step = self.optimal_time_step

    def satisfy_cfl(self) -> bool:
        return self.desired_time_step < self.optimal_time_step + 1e-8
