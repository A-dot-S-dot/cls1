from typing import Callable

import numpy as np
import pde_solver.system_matrix as matrix
from pde_solver.discretization.finite_element import (
    LagrangeFiniteElementSpace,
    QuadratureFastElement,
)
from pde_solver.system_vector.flux_approximation import FluxApproximation

from .discrete_l2_product import BasisGradientL2ProductEntryCalculator
from .local_vector import LocallyAssembledVector
from .system_vector import SystemVector

ScalarFunction = Callable[[float], float]


class FluxGradientEntryCalculator(BasisGradientL2ProductEntryCalculator):
    dof_vector: np.ndarray

    _flux: ScalarFunction
    _fast_element: QuadratureFastElement

    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
        flux: ScalarFunction,
    ):
        BasisGradientL2ProductEntryCalculator.__init__(
            self, element_space, 2**element_space.polynomial_degree
        )
        self._flux = flux
        self._build_finite_element()

    def _build_finite_element(self):
        self._fast_element = QuadratureFastElement(
            self._element_space, self._local_quadrature
        )
        self._fast_element.set_values()

    def set_dofs(self, dof_vector: np.ndarray):
        self._fast_element.dof_vector = dof_vector

    def left_function(self, simplex_index: int, quadrature_node_index: int) -> float:
        finite_element_value = self._fast_element(simplex_index, quadrature_node_index)

        return self._flux(finite_element_value)


class FluxGradient(LocallyAssembledVector):
    """Flux derivative vector. It's entries are

        f(v) * phi'_i,

    where {phi_i}_i denotes the basis of the element space, f a flux and v a
    finite element.

    Not assembled by default.

    """

    _entry_calculator: FluxGradientEntryCalculator

    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
        flux: ScalarFunction,
    ):
        self._entry_calculator = FluxGradientEntryCalculator(element_space, flux)

        LocallyAssembledVector.__init__(self, element_space, self._entry_calculator)

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        self._entry_calculator.set_dofs(dof_vector)
        return super().__call__()


class AdvectionFluxGradient(SystemVector):
    """Advection flux gradeint vector. It's entries are

        sum((bj,Dbi) * uj),

    where bi denotes the basis of the element space.

    Not assembled by default.

    """

    _discrete_gradient: matrix.DiscreteGradient

    def __init__(self, element_space: LagrangeFiniteElementSpace):
        self._discrete_gradient = matrix.DiscreteGradient(element_space)

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return -self._discrete_gradient.dot(dof_vector)


class ApproximatedFluxGradient(SystemVector):
    """Approximated flux derivative vector. It's entries are

        F * Dbi,

    where bi denotes the basis of the element space and F the GFE approximation.

    Not assembled by default.

    """

    _discrete_gradient: matrix.DiscreteGradient
    _flux_approixmation: SystemVector

    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
        flux: Callable[[np.ndarray], np.ndarray],
    ):
        self._discrete_gradient = matrix.DiscreteGradient(element_space)
        self._flux_approximation = FluxApproximation(flux)

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return -self._discrete_gradient.dot(self._flux_approximation(dof_vector))
