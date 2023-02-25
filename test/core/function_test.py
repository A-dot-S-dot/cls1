from unittest import TestCase
from core.function import *
from numpy.testing import assert_equal


class TestMaximum(TestCase):
    def test_function(self):
        max = maximum([1.0, 2.0], [2.0, -1.0], [0.0, 3.0])
        expected_output = np.array([2.0, 3.0])

        assert_equal(max, expected_output)


class TestMinimum(TestCase):
    def test_function(self):
        min = minimum([1.0, 2.0], [2.0, -1.0], [0.0, 3.0])
        expected_output = np.array([0.0, -1.0])

        assert_equal(min, expected_output)
