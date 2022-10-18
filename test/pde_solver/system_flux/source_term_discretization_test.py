from unittest import TestCase

from pde_solver.system_flux import (
    NaturalSourceTermDiscretization,
    WetDryPreservingSourceTermDiscretization,
)


class TestNaturalSourceTermDiscretization(TestCase):
    discretization = NaturalSourceTermDiscretization()
    discretization.step_length = 0.5
    left_height = [1]
    right_height = [4]
    topography_step = [1]
    expected_source_term = [5]

    def test_numerical_flux(self):
        for i in range(len(self.expected_source_term)):
            self.assertAlmostEqual(
                self.discretization(
                    self.left_height[i], self.right_height[i], self.topography_step[i]
                ),
                self.expected_source_term[i],
                msg=f"right flux, index={i}",
            )


class TestWetDryPreservingSourceTermDiscretization(TestNaturalSourceTermDiscretization):
    discretization = WetDryPreservingSourceTermDiscretization()
    discretization.step_length = 0.5
    left_height = [1, 1]
    right_height = [4, 4]
    topography_step = [1, -1]
    expected_source_term = [5, -5]
