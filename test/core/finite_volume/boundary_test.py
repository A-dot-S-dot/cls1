from unittest import TestCase

from core.finite_volume.boundary import *
from numpy.testing import assert_equal


class TestBoundaryConditionApplier(BoundaryConditionApplier):
    def add_condition(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        return (
            np.array([time, *dof_vector])
            if self.left
            else np.array([*dof_vector, time])
        )


class TestBoundaryMethods(TestCase):
    def test_no_side_error(self):
        self.assertRaises(ValueError, TestBoundaryConditionApplier, "0")
        self.assertRaises(ValueError, TestBoundaryConditionApplier, "LEFT")
        self.assertRaises(ValueError, TestBoundaryConditionApplier, "Left")

    def test_left(self):
        boundary = TestBoundaryConditionApplier("left")
        self.assertTrue(boundary.left)
        self.assertFalse(boundary.right)

    def test_right(self):
        boundary = TestBoundaryConditionApplier("right")
        self.assertTrue(boundary.right)
        self.assertFalse(boundary.left)


class TestOutflowBoundary(TestCase):
    def test_left_scalar(self):
        vector = np.array([1, 2, 3, 4])
        expected_output = np.array([1, 1, 1, 2, 3, 4])

        condition_applier = OutflowConditionApplier("left", cells_to_add_number=2)
        self.assertTrue(
            np.array_equal(condition_applier.add_condition(1, vector), expected_output)
        )

    def test_right_scalar(self):
        vector = np.array([1, 2, 3, 4])
        expected_output = np.array([1, 2, 3, 4, 4])

        condition_applier = OutflowConditionApplier("right")
        self.assertTrue(
            np.array_equal(condition_applier.add_condition(1, vector), expected_output)
        )

    def test_left_system(self):
        vector = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected_output = np.array([[1, 2], [1, 2], [3, 4], [5, 6], [7, 8]])

        condition_applier = OutflowConditionApplier("left")
        self.assertTrue(
            np.array_equal(condition_applier.add_condition(1, vector), expected_output)
        )

    def test_right_system(self):
        vector = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected_output = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [7, 8]])

        condition_applier = OutflowConditionApplier("right")
        self.assertTrue(
            np.array_equal(condition_applier.add_condition(1, vector), expected_output)
        )


class TestInflowBounday(TestCase):
    def test_left_scalar(self):
        condition = lambda t: t
        condition_applier = InflowConditionApplier("left", condition)

        vector = np.array([1, 2, 3, 4])
        expected_output = np.array([1, 1, 2, 3, 4])
        self.assertTrue(
            np.array_equal(condition_applier.add_condition(1, vector), expected_output)
        )

    def test_right_scalar(self):
        condition = lambda t: t
        condition_applier = InflowConditionApplier("right", condition)

        vector = np.array([1, 2, 3, 4])
        expected_output = np.array([1, 2, 3, 4, 1])

        self.assertTrue(
            np.array_equal(condition_applier.add_condition(1, vector), expected_output)
        )

    def test_left_system(self):
        condition = lambda t: np.array([t, 2 * t])
        condition_applier = InflowConditionApplier("left", condition)

        vector = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected_output = np.array([[1, 2], [1, 2], [3, 4], [5, 6], [7, 8]])

        self.assertTrue(
            np.array_equal(condition_applier.add_condition(1, vector), expected_output)
        )

    def test_right_system(self):
        condition = lambda t: np.array([t, 2 * t])
        condition_applier = InflowConditionApplier("right", condition)

        vector = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected_output = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [1, 2]])

        self.assertTrue(
            np.array_equal(condition_applier.add_condition(1, vector), expected_output)
        )


class TestBoundaryCondition(TestCase):
    def test_not_right_side_error(self):
        self.assertRaises(
            ValueError,
            BoundaryConditionsApplier,
            TestBoundaryConditionApplier("right"),
            TestBoundaryConditionApplier("right"),
        )

    def test_condition(self):
        conditions_applier = BoundaryConditionsApplier(
            TestBoundaryConditionApplier("left"), TestBoundaryConditionApplier("right")
        )
        vector = np.array([1, 2])

        expected_output = np.array([0, 1, 2, 0])

        self.assertTrue(
            np.array_equal(
                conditions_applier.add_conditions(0, vector), expected_output
            )
        )


class TestPeriodicBoundaryCondition(TestCase):
    def test_scalar_input(self):
        conditions_applier = PeriodicBoundaryConditionsApplier()
        vector = np.array([1, 2])

        expected_output = np.array([2, 1, 2, 1])
        self.assertTrue(
            np.array_equal(
                conditions_applier.add_conditions(0, vector), expected_output
            )
        )

    def test_system_input(self):
        conditions_applier = PeriodicBoundaryConditionsApplier()
        vector = np.array([[1.0, 2.0], [3.0, 4.0]])

        expected_output = np.array([[3.0, 4.0], [1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])
        self.assertTrue(
            np.array_equal(
                conditions_applier.add_conditions(0, vector), expected_output
            )
        )

    def test_higher_cell_to_add_numbers(self):
        conditions_applier = PeriodicBoundaryConditionsApplier((2, 3))
        vector = np.array([1, 2, 3, 4, 5, 6, 7, 8])

        expected_output = np.array([7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3])

        assert_equal(conditions_applier.add_conditions(0, vector), expected_output)
