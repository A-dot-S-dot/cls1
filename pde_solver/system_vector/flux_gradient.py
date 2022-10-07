from typing import Callable
import numpy as np
from custom_type import ScalarFunction
from pde_solver.discretization.finite_element import QuadratureFastFiniteElement
from pde_solver.solver_space import LagrangeFiniteElementSpace
from pde_solver.system_matrix import SystemMatrix
from pde_solver.system_vector.flux_approximation import FluxApproximation

from .discrete_l2_product import BasisGradientL2ProductEntryCalculator
from .local_vector import LocallyAssembledVector
from .system_vector import SystemVector


class FluxGradientEntryCalculator(BasisGradientL2ProductEntryCalculator):
    dof_vector: np.ndarray

    _flux: ScalarFunction
    _fast_element: QuadratureFastFiniteElement

    def __init__(
        self,
        element_space: LagrangeFiniteElementSpace,
        quadrature_degree: int,
        flux: ScalarFunction,
    ):
        BasisGradientL2ProductEntryCalculator.__init__(
            self, element_space, quadrature_degree
        )
        self._flux = flux
        self._build_finite_element()

    def _build_finite_element(self):
        self._fast_element = QuadratureFastFiniteElement(
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
        quadrature_degree: int,
        flux: ScalarFunction,
    ):
        self._entry_calculator = FluxGradientEntryCalculator(
            element_space, quadrature_degree, flux
        )

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

    _discrete_gradient: SystemMatrix

    def __init__(
        self,
        discrete_gradient: SystemMatrix,
    ):
        self._discrete_gradient = discrete_gradient

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return self._discrete_gradient.dot(dof_vector)


class ApproximatedFluxGradient(SystemVector):
    """Approximated flux derivative vector. It's entries are

        F * Dbi,

    where bi denotes the basis of the element space and F the GFE approximation.

    Not assembled by default.

    """

    _discrete_gradient: SystemMatrix
    _flux_approixmation: SystemVector

    def __init__(
        self,
        discrete_gradient: SystemMatrix,
        flux: Callable[[np.ndarray], np.ndarray],
    ):
        self._discrete_gradient = discrete_gradient
        self._flux_approximation = FluxApproximation(flux)

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return -self._discrete_gradient.dot(self._flux_approximation(dof_vector))
