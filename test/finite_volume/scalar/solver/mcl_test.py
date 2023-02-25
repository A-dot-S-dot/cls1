from unittest import TestCase
from finite_volume.scalar.solver.mcl import *
from finite_volume import scalar
import finite_volume


class TestMCL(TestCase):
    def test_flux(self):
        flux = MCLFlux(
            scalar.get_riemann_solver("advection"),
            finite_volume.NeighbourIndicesMapping(4, True),
        )
        print(flux(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 0.4])))
