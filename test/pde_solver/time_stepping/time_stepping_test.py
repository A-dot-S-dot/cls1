from test.test_helper import QUADRATIC_MESH
from typing import List
from unittest import TestCase


from pde_solver.time_stepping import *


class TestSpatialMeshDependentTimeStepping(TestCase):
    time_stepping = SpatialMeshDependendetTimeStepping()
    time_stepping.start_time = 0
    time_stepping.end_time = 1
    time_stepping.mesh = QUADRATIC_MESH

    def test_uniform_time_stepping(self):
        self._test_time_stepping(1, [0.5, 0.5])

    def _test_time_stepping(
        self, cfl_number: float, expected_time_stepping: List[float]
    ):
        self.time_stepping.cfl_number = cfl_number
        time_stepping = [self.time_stepping.time_step for _ in self.time_stepping]
        self.assertListEqual(time_stepping, expected_time_stepping)

    def test_not_uniform_time_stepping(self):
        self._test_time_stepping(0.75, [0.375, 0.375, 0.25])

    def test_to_small_break(self):
        self.assertRaises(TimeStepTooSmallError, self._test_time_stepping, 1e-13, [])
