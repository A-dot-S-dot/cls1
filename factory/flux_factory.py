from system.matrix import SystemMatrix
from system.vector import SystemVector
from system.vector.dof_vector import DOFVector
from system.vector.flux_gradient import (
    AdvectionFluxGradient,
    ApproximatedFluxGradient,
    FluxGradient,
)
from system.vector.group_finite_element_approximation import (
    GroupFiniteElementApproximation,
)


class FluxFactory:
    problem_name: str
    exact_flux = False

    def get_flux_gradient(
        self,
        dof_vector: DOFVector,
        discrete_gradient: SystemMatrix,
    ) -> SystemVector:
        if self.problem_name == "advection":
            return AdvectionFluxGradient(dof_vector, discrete_gradient)

        else:
            flux = self.flux

            if self.exact_flux:
                return FluxGradient(
                    dof_vector, flux, 2 * dof_vector.element_space.polynomial_degree
                )
            else:
                flux_approximation = GroupFiniteElementApproximation(dof_vector, flux)
                return ApproximatedFluxGradient(flux_approximation, discrete_gradient)

    @property
    def flux(self):
        if self.problem_name == "advection":
            return lambda u: u
        elif self.problem_name == "burgers":
            return lambda u: 1 / 2 * u**2
        else:
            raise NotImplementedError(f"No flux for '{self.problem_name}' implemented.")
