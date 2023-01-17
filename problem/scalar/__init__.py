from .artificial_diffusion import (
    BurgersArtificialDiffusion,
    DiscreteUpwind,
    build_artificial_diffusion,
)
from .discrete_gradient import DiscreteGradient
from .discrete_l2_product import BasisGradientL2Product, BasisL2Product
from .flux_approximation import FluxApproximation, build_flux_approximation
from .flux_gradient import (
    AdvectionFluxGradient,
    ApproximatedFluxGradient,
    FluxGradient,
    build_exact_flux_gradient,
    build_flux_gradient,
    build_flux_gradient_approximation,
)
from .local_bounds import LocalMaximum, LocalMinimum
from .lumped_mass import LumpedMassVector
from .mass import MassMatrix
