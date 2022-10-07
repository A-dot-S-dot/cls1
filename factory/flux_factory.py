import numpy as np
from pde_solver.solver_space import LagrangeFiniteElementSpace
from pde_solver.system_matrix import SystemMatrix
from pde_solver.system_vector import (
    AdvectionFluxGradient,
    ApproximatedFluxGradient,
    FluxGradient,
    SystemVector,
)


class FluxFactory:
    element_space: LagrangeFiniteElementSpace
    problem_name: str
    exact_flux = False

    def get_flux_gradient(self, discrete_gradient: SystemMatrix) -> SystemVector:
        if self.problem_name == "advection":
            return AdvectionFluxGradient(discrete_gradient)

        else:
            return self._non_advection_flux_gradient(discrete_gradient)

    def _non_advection_flux_gradient(self, discrete_gradient: SystemMatrix):
        if self.exact_flux:
            return FluxGradient(
                self.element_space, 2 * self.element_space.polynomial_degree, self.flux
            )
        else:
            return ApproximatedFluxGradient(discrete_gradient, self.flux)

    @property
    def flux(self):
        if self.problem_name == "advection":
            return lambda u: u
        elif self.problem_name == "burgers":
            return lambda u: 1 / 2 * u**2
        else:
            raise NotImplementedError(f"No flux for '{self.problem_name}' implemented.")


FLUX_FACTORY = FluxFactory()
