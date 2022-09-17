from mesh import Interval

from .gauss import GaussianQuadratureGeneralized


class LocalElementQuadrature(GaussianQuadratureGeneralized):
    """Gaussian Quadrature on the standard element [0,1]."""

    def __init__(self, nodes_number: int):
        """The quadrature is 2*NODES_NUMBER-1 exact."""
        GaussianQuadratureGeneralized.__init__(self, nodes_number, Interval(0, 1))
