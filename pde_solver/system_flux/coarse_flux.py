from typing import Tuple

import numpy as np
from pde_solver.discrete_solution import DiscreteSolution
from pde_solver.solver_space import FiniteVolumeSpace
from pde_solver.time_stepping import TimeStepping

from .system_flux import SystemFlux


class FlatBottomCoarseFlux(SystemFlux):
    """We consider a fine solution and a coarse one, which is created by
    averaging the fine ones.

    Because of CFL issues we calculate Un+1 by calculating uk for several k
    until the fine solution reaches the time tn+DT, where DT is the time step
    for the coarse solution. The resulting flux is calculated here. This
    calculator is designed for Shallow water problems and assumes a flat
    boundary, since in this case we have a conservation law and left and right
    fluxes are equal. Then we obtain the following flux

        FL = 1/(N*DX*DT)*sum(dt_l*fL(ul)),
        FR = 1/(N*DX*DT)*sum(dt_l*fR(ul)),

    where N denotes the coarsening degre, DX the mesh size of the coarse grid,
    DT the coarse time step and dt_l the l-th fine time step.

    """

    coarse_volume_space: FiniteVolumeSpace
    coarsening_degree: float
    fine_solution: DiscreteSolution
    fine_numerical_flux: SystemFlux
    fine_step_length: float

    _fine_time_stepping: TimeStepping

    @property
    def fine_time_stepping(self) -> TimeStepping:
        return self._fine_time_stepping

    @fine_time_stepping.setter
    def fine_time_stepping(self, time_stepping: TimeStepping):
        time_stepping.end_time = time_stepping.start_time
        self._fine_time_stepping = time_stepping

    def __call__(self, time_step: float) -> Tuple[np.ndarray, np.ndarray]:
        left_flux = np.zeros((self.coarse_volume_space.edge_number, 2))
        right_flux = np.zeros((self.coarse_volume_space.edge_number, 2))
        factor = 1 / (self.coarsening_degree * self.fine_step_length * time_step)

        self.fine_time_stepping.end_time += time_step

        for _ in self.fine_time_stepping:
            fine_time_step = self.fine_time_stepping.time_step
            left_subflux, right_subflux = self._calcualte_temporal_subflux(
                fine_time_step
            )
            left_flux += fine_time_step * left_subflux
            right_flux += fine_time_step * right_subflux

        return (factor * left_flux, factor * right_flux)

    def _calcualte_temporal_subflux(
        self, fine_time_step: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_fine_flux, right_fine_flux = self.fine_numerical_flux(
            self.fine_solution.end_values
        )
        self._update_solution(fine_time_step, left_fine_flux, right_fine_flux)
        return (
            left_fine_flux[:: self.coarsening_degree],
            right_fine_flux[self.coarsening_degree - 1 :: self.coarsening_degree],
        )

    def _update_solution(
        self,
        fine_time_step: float,
        left_fine_flux: np.ndarray,
        right_fine_flux: np.ndarray,
    ):
        updated_solution = (
            self.fine_solution.end_values
            + fine_time_step
            * (right_fine_flux + -left_fine_flux)
            / self.fine_step_length
        )
        self.fine_solution.add_solution(fine_time_step, updated_solution)
