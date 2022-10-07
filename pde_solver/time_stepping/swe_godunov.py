import numpy as np
from pde_solver.mesh import Mesh
from pde_solver.system_vector import SWEGodunovNumericalFlux


from .time_stepping import TimeStepping


class SWEGodunovTimeStepping(TimeStepping):
    """This time stepping ensures for each time step

        dt = CFL_NUMBER*min(mi/dii)

    where mi denotes the lumped mass and dij a diffusion. At the end dt could be
    smaller to reach end time.

    """

    mesh: Mesh
    numerical_flux: SWEGodunovNumericalFlux

    @property
    def desired_time_step(self) -> float:
        return (
            self.cfl_number
            * self.mesh.step_length
            / (
                2
                * np.max(
                    [
                        abs(self.numerical_flux.left_intermediate_velocity),
                        self.numerical_flux.right_intermediate_velocity,
                    ]
                )
            )
        )
