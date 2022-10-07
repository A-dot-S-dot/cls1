from pde_solver import PDESolver
from pde_solver.mesh import Mesh
from pde_solver.system_vector import SWEGodunovNumericalFlux
from pde_solver.time_stepping import SWEGodunovTimeStepping
from tqdm import tqdm


class SWEGodunovSolver(PDESolver):
    """Solves Shallow water equation with Godunovs method."""

    time_stepping: SWEGodunovTimeStepping
    numerical_flux: SWEGodunovNumericalFlux
    mesh: Mesh

    def solve(self):
        progress_iterator = tqdm(self.time_stepping, **self.tqdm_kwargs)

        for _ in progress_iterator:
            self.update()

    def update(self):
        flux = self.numerical_flux(self.solution.end_solution)
        time_step = self.time_stepping.time_step

        updated_solution = (
            self.solution.end_solution + time_step * flux / self.mesh.step_length
        )
        self.solution.add_solution(time_step, updated_solution)
