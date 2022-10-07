from pde_solver.discrete_solution import DiscreteSolutionObservable
from pde_solver.system_matrix import (
    BurgersArtificialDiffusion,
    DiscreteUpwind,
    SystemMatrix,
)


class ArtificialDiffusionFactory:
    problem_name: str

    def get_artificial_diffusion(
        self,
        discrete_gradient: SystemMatrix,
        discrete_solution: DiscreteSolutionObservable,
    ) -> SystemMatrix:
        if self.problem_name == "advection":
            return DiscreteUpwind(discrete_gradient)
        elif self.problem_name == "burgers":
            diffusion = BurgersArtificialDiffusion(discrete_gradient, discrete_solution)
            diffusion.update()
            return diffusion
        else:
            raise NotImplementedError(
                f"No Artificial diffusion for {self.problem_name} available."
            )


DIFFUSION_FACTORY = ArtificialDiffusionFactory()
