from pde_solver import PDESolver
from pde_solver.mesh import Mesh
from pde_solver.system_vector import SWEGodunovNumericalFlux
from pde_solver.time_stepping import SWEGodunovTimeStepping


class SWEGodunovSolver(PDESolver):
    """Solves Shallow water equation with Godunovs method."""

    time_stepping: SWEGodunovTimeStepping
    numerical_flux: SWEGodunovNumericalFlux
    mesh: Mesh

    def update(self):
        left_flux, right_flux = self.numerical_flux(self.solution.end_solution)
        time_step = self.time_stepping.time_step

        updated_solution = (
            self.solution.end_solution
            - time_step * (left_flux - right_flux) / self.mesh.step_length
        )
        self.solution.add_solution(time_step, updated_solution)
