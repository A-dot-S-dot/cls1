from system.matrix import SystemMatrix
from system.matrix.artificial_diffusion import (
    BurgersArtificialDiffusion,
    DiscreteUpwind,
)
from system.vector.dof_vector import DOFVector


class ArtificialDiffusionFactory:
    problem_name: str

    def get_artificial_diffusion(
        self,
        dof_vector: DOFVector,
        discrete_gradient: SystemMatrix,
    ) -> SystemMatrix:
        if self.problem_name == "advection":
            return DiscreteUpwind(discrete_gradient)
        elif self.problem_name == "burgers":
            return BurgersArtificialDiffusion(dof_vector, discrete_gradient)
        else:
            raise NotImplementedError(
                f"No Artificial diffusion for {self.problem_name} available."
            )
