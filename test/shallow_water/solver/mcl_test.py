from unittest import TestCase

import numpy as np
from shallow_water.solver.mcl import *
import lib
import core


class TestMCLFlux(TestCase):
    values_left = np.array(
        [[2.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0]]
    )
    values_right = np.array(
        [[1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [2.0, 0.0], [1.0, 1.0]]
    )
    flux = MCLFlux(
        1,
        lib.CentralFlux(swe.Flux(1)),
        core.get_boundary_conditions("periodic"),
    )
    flux_left, flux_right = flux(values_left, values_right)
