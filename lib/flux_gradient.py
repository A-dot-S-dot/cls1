from typing import Callable

import numpy as np
from core import finite_element

from .discrete_gradient import DiscreteGradient
from .discrete_l2_product import BasisGradientL2ProductEntryCalculator
from .flux_approximation import FluxApproximation


class FluxGradientEntryCalculator(BasisGradientL2ProductEntryCalculator):
    dof_vector: np.ndarray

    _flux: Callable[[float], float]
    _fast_element: finite_element.QuadratureFastElement

    def __init__(
        self,
        element_space: finite_element.LagrangeSpace,
        flux: Callable[[float], float],
    ):
        BasisGradientL2ProductEntryCalculator.__init__(
            self, element_space, 2**element_space.polynomial_degree
        )
        self._flux = flux
        self._build_finite_element()

    def _build_finite_element(self):
        self._fast_element = finite_element.QuadratureFastElement(
            self._space, self._local_quadrature
        )
        self._fast_element.set_values()

    def set_dofs(self, dof_vector: np.ndarray):
        self._fast_element.dof_vector = dof_vector

    def left_function(self, simplex_index: int, quadrature_node_index: int) -> float:
        finite_element_value = self._fast_element(simplex_index, quadrature_node_index)

        return self._flux(finite_element_value)


class FluxGradient(finite_element.LocallyAssembledVector):
    """Flux derivative vector. It's entries are

        f(v) * phi'_i,

    where {phi_i}_i denotes the basis of the element space, f a flux and v a
    finite element.

    Not assembled by default.

    """

    _entry_calculator: FluxGradientEntryCalculator

    def __init__(
        self,
        element_space: finite_element.LagrangeSpace,
        flux: Callable[[float], float],
    ):
        self._entry_calculator = FluxGradientEntryCalculator(element_space, flux)

        finite_element.LocallyAssembledVector.__init__(
            self, element_space, self._entry_calculator
        )

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        self._entry_calculator.set_dofs(dof_vector)
        return super().__call__()


class AdvectionFluxGradient:
    """Advection flux gradeint vector. It's entries are

        sum((bj,Dbi) * uj),

    where bi denotes the basis of the element space.

    Not assembled by default.

    """

    _discrete_gradient: DiscreteGradient

    def __init__(self, element_space: finite_element.LagrangeSpace):
        self._discrete_gradient = DiscreteGradient(element_space)

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return -self._discrete_gradient.dot(dof_vector)


class ApproximatedFluxGradient:
    """Approximated flux derivative vector. It's entries are

        F * Dbi,

    where bi denotes the basis of the element space and F the GFE approximation.

    Not assembled by default.

    """

    _discrete_gradient: DiscreteGradient
    _flux_approximation: Callable[[np.ndarray], np.ndarray]

    def __init__(
        self,
        element_space: finite_element.LagrangeSpace,
        flux: Callable[[np.ndarray], np.ndarray],
    ):
        self._discrete_gradient = DiscreteGradient(element_space)
        self._flux_approximation = FluxApproximation(flux)

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return -self._discrete_gradient.dot(self._flux_approximation(dof_vector))


def build_exact_flux_gradient(
    problem: str, element_space: finite_element.LagrangeSpace
) -> Callable[[np.ndarray], np.ndarray]:
    flux_gradients = {
        "advection": AdvectionFluxGradient(element_space),
        "burgers": FluxGradient(element_space, lambda u: 1 / 2 * u**2),
    }
    return flux_gradients[problem]


def build_flux_gradient_approximation(
    problem: str, element_space: finite_element.LagrangeSpace
) -> Callable[[np.ndarray], np.ndarray]:
    flux_gradients = {
        "advection": AdvectionFluxGradient(element_space),
        "burgers": ApproximatedFluxGradient(element_space, lambda u: 1 / 2 * u**2),
    }
    return flux_gradients[problem]


def build_flux_gradient(
    problem: str, element_space: finite_element.LagrangeSpace, exact_flux: bool
) -> Callable[[np.ndarray], np.ndarray]:
    if exact_flux:
        return build_exact_flux_gradient(problem, element_space)
    else:
        return build_flux_gradient_approximation(problem, element_space)
