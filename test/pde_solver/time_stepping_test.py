from unittest import TestCase

from pde_solver.time_stepping import SpatialMeshDependendetTimeStepping

from ..test_helper import QUADRATIC_MESH


class TestSpatialMeshDependentTimeStepping(TestCase):
    def test_uniform_time_stepping(self):
        start_time = 0
        end_time = 1
        cfl_number = 1
        time_stepping = SpatialMeshDependendetTimeStepping(
            start_time, end_time, QUADRATIC_MESH, cfl_number
        )
        expected_time_stepping = [0.5, 0.5]

        for i, delta_t in enumerate(time_stepping):
            expected_delta_t = expected_time_stepping[i]
            self.assertAlmostEqual(delta_t, expected_delta_t, msg=f"index={i}")

    def test_not_uniform_time_stepping(self):
        start_time = 0
        end_time = 1
        courant_factor = 0.75
        time_stepping = SpatialMeshDependendetTimeStepping(
            start_time, end_time, QUADRATIC_MESH, courant_factor
        )
        expected_time_stepping = [0.375, 0.375, 0.25]

        for i, delta_t in enumerate(time_stepping):
            expected_delta_t = expected_time_stepping[i]

            self.assertAlmostEqual(delta_t, expected_delta_t, msg=f"index={i}")
