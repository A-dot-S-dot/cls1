from unittest import TestCase

from parser.types import positive_float, positive_int, percent_number


class TestPositiveInt(TestCase):
    def test_value_errors(self):
        self.assertRaises(ValueError, positive_float, "-2")
        self.assertRaises(ValueError, positive_float, "0")

    def test_types(self):
        self.assertRaises(ValueError, positive_float, "1 + 3j")
        self.assertRaises(ValueError, positive_float, "True")
        self.assertRaises(ValueError, positive_float, "weird string")


class TestPositiveFloat(TestCase):
    def test_value_errors(self):
        self.assertRaises(ValueError, positive_int, "-2")
        self.assertRaises(ValueError, positive_int, "0")

    def test_types(self):
        self.assertRaises(ValueError, positive_int, "1 + 3j")
        self.assertRaises(ValueError, positive_int, "True")
        self.assertRaises(ValueError, positive_int, "weird string")


class TestPositiveRatioNumber(TestCase):
    def test_value_errors(self):
        self.assertRaises(ValueError, percent_number, "-2")
        self.assertRaises(ValueError, percent_number, "1.1")

    def test_types(self):
        self.assertRaises(ValueError, percent_number, "1 + 3j")
        self.assertRaises(ValueError, percent_number, "True")
        self.assertRaises(ValueError, percent_number, "weird string")
