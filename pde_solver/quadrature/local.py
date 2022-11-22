from pde_solver.mesh import Interval

from .gauss import GaussianQuadrature


class LocalElementQuadrature(GaussianQuadrature):
    """Gaussian Quadrature on the standard element [0,1]."""

    def __init__(self, quadrature_degree: int):
        """The quadrature is 2*NODES_NUMBER-1 exact."""
        GaussianQuadrature.__init__(self, quadrature_degree, Interval(0, 1))
