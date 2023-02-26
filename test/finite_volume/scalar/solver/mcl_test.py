from unittest import TestCase, skip

import finite_volume
from finite_volume import scalar
from finite_volume.scalar.solver.mcl import *


class TestMCL(TestCase):
    @skip("No test implemented.")
    def test_flux(self):
        flux = MCLFlux(
            scalar.get_riemann_solver("advection"),
            finite_volume.NeighbourIndicesMapping(4, True),
        )
