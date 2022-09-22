from fem.fast_element import QuadratureFastFiniteElement
from math_type import FunctionRealToReal
from system.matrix import SystemMatrix

from .discrete_l2_product import AbstractBasisGradientL2ProductEntryCalculator
from .dof_vector import DOFVector
from .system_vector import LocallyAssembledSystemVector, SystemVector


class FluxGradientEntryCalculator(AbstractBasisGradientL2ProductEntryCalculator):
    _flux: FunctionRealToReal
    _fast_element: QuadratureFastFiniteElement

    def __init__(
        self,
        dof_vector: DOFVector,
        flux: FunctionRealToReal,
        quadrature_degree: int,
    ):
        AbstractBasisGradientL2ProductEntryCalculator.__init__(
            self, dof_vector.element_space, quadrature_degree
        )

        self._flux = flux

        self._build_finite_element(dof_vector)

    def _build_finite_element(self, dof_vector: DOFVector):
        self._fast_element = QuadratureFastFiniteElement(
            self._element_space, self._local_quadrature
        )
        self._fast_element.set_values()
        self._fast_element.dofs = dof_vector

    def _left_function(self, simplex_index: int, quadrature_node_index: int) -> float:
        finite_element_value = self._fast_element.value(
            simplex_index, quadrature_node_index
        )

        return self._flux(finite_element_value)


class FluxGradient(LocallyAssembledSystemVector):
    """Flux derivative vector. It's entries are

        f(v) * phi'_i,

    where {phi_i}_i denotes the basis of the element space, f a flux and v a
    finite element.

    Not assembled by default.

    """

    def __init__(
        self,
        dof_vector: DOFVector,
        flux: FunctionRealToReal,
        quadrature_degree: int,
    ):
        entry_calculator = FluxGradientEntryCalculator(
            dof_vector, flux, quadrature_degree
        )
        dof_vector.register_observer(self)

        LocallyAssembledSystemVector.__init__(
            self, dof_vector.element_space, entry_calculator
        )


class AdvectionFluxGradient(SystemVector):
    """Advection flux gradeint vector. It's entries are

        sum((bj,Dbi) * uj),

    where bi denotes the basis of the element space.

    Not assembled by default.

    """

    _dof_vector: DOFVector
    _discrete_gradient: SystemMatrix

    def __init__(
        self,
        dof_vector: DOFVector,
        discrete_gradient: SystemMatrix,
    ):
        SystemVector.__init__(self, dof_vector.element_space)
        self._discrete_gradient = discrete_gradient
        self._dof_vector = dof_vector
        dof_vector.register_observer(self)

    def assemble(self):
        self[:] = self._discrete_gradient.dot(self._dof_vector)


class ApproximatedFluxGradient(SystemVector):
    """Approximated flux derivative vector. It's entries are

        F * Dbi,

    where bi denotes the basis of the element space and F the GFE approximation.

    Not assembled by default.

    """

    def __init__(
        self,
        flux_approximation: DOFVector,
        discrete_gradient: SystemMatrix,
    ):
        SystemVector.__init__(self, flux_approximation.element_space)
        self._discrete_gradient = discrete_gradient
        self._flux_approximation = flux_approximation

        flux_approximation.register_observer(self)

    def assemble(self):
        self._values = -self._discrete_gradient.dot(self._flux_approximation)
