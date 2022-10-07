from unittest import TestCase

from pde_solver.mesh import Interval


class TestInterval(TestCase):
    interval = Interval(0, 1)

    def test_type_errors(self):
        self.assertRaises(ValueError, Interval, "a", "b")
        self.assertRaises(ValueError, Interval, 1, "b")
        self.assertRaises(ValueError, Interval, "a", 2)

    def test_wrong_order_error(self):
        self.assertRaises(AssertionError, Interval, 1, 0)

    def test_equal_error(self):
        self.assertRaises(AssertionError, Interval, 0, 0)

    def test_a_getter(self):
        self.assertEqual(self.interval.a, 0)

    def test_b_getter(self):
        self.assertEqual(self.interval.b, 1)

    def test_length(self):
        self.assertAlmostEqual(self.interval.length, 1.0, 12)

    def test_is_equal(self):
        other_interval = Interval(0, 1)
        other_interval_2 = Interval(-1, 1)

        self.assertTrue(self.interval == other_interval)
        self.assertTrue(self.interval != other_interval_2)

    def test_contains(self):
        inner_points = [0.1, 0.5, 0.75, 0.99999]
        boundary_points = [0, 1]
        outer_points = [-2, 1000, 1.00000001]

        for point in inner_points:
            self.assertTrue(point in self.interval)

        for point in boundary_points:
            self.assertTrue(point in self.interval)

        for point in outer_points:
            self.assertFalse(point in self.interval)

    def test_is_in_inner(self):
        inner_points = [0.1, 0.5, 0.75, 0.99999]
        boundary_points = [0, 1]
        outer_points = [-2, 1000, 1.00000001]

        for point in inner_points:
            self.assertTrue(self.interval.is_in_inner(point))

        for point in boundary_points:
            self.assertFalse(self.interval.is_in_inner(point))

        for point in outer_points:
            self.assertFalse(self.interval.is_in_inner(point))

    def test_is_in_boundary(self):
        inner_points = [0.1, 0.5, 0.75, 0.99999]
        boundary_points = [0, 1]
        outer_points = [-2, 1000, 1.00000001]

        for point in inner_points:
            self.assertFalse(self.interval.is_in_boundary(point))

        for point in boundary_points:
            self.assertTrue(self.interval.is_in_boundary(point))

        for point in outer_points:
            self.assertFalse(self.interval.is_in_boundary(point))
