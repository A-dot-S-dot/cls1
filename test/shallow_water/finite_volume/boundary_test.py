from unittest import TestCase

import numpy as np

from shallow_water.finite_volume.boundary import *


class TestReflectingBoundayConditionApplier(TestCase):
    def test_left(self):
        condition_applier = ReflectingCondition("left")

        vector = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected_output = np.array([[1, -2], [1, 2], [3, 4], [5, 6], [7, 8]])

        self.assertTrue(
            np.array_equal(condition_applier.add_condition(1, vector), expected_output)
        )

    def test_right(self):
        condition_applier = ReflectingCondition("right")

        vector = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected_output = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [7, -8]])

        self.assertTrue(
            np.array_equal(condition_applier.add_condition(1, vector), expected_output)
        )
