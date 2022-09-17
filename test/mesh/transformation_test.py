from unittest import TestCase

import numpy as np
from ..test_helper import discrete_derivative

from mesh import Interval
from mesh.transformation import AffineTransformation


class TestAffineTransformation(TestCase):
    simplex = Interval(-1, 1)
    simplex_points = [-1, 0, 1]
    standard_simplex_points = [0, 0.5, 1]
    affine_transformation = AffineTransformation()

    def test_call(self):
        for x, x_standard in zip(self.simplex_points, self.standard_simplex_points):
            self.assertEqual(self.affine_transformation(x_standard, self.simplex), x)

    def test_inverse(self):
        for x, x_standard in zip(self.simplex_points, self.standard_simplex_points):
            self.assertEqual(
                self.affine_transformation.inverse(x, self.simplex), x_standard
            )

    def test_inverse_property(self):
        for x, x_standard in zip(self.simplex_points, self.standard_simplex_points):
            self.assertEqual(
                self.affine_transformation.inverse(
                    self.affine_transformation(x, self.simplex), self.simplex
                ),
                x,
            )
            self.assertEqual(
                self.affine_transformation(
                    self.affine_transformation.inverse(x_standard, self.simplex),
                    self.simplex,
                ),
                x_standard,
            )

    def test_derivative(self):
        for x in np.linspace(self.simplex.a, self.simplex.b):
            self.assertAlmostEqual(
                self.affine_transformation.derivative(self.simplex),
                discrete_derivative(
                    lambda x: self.affine_transformation(x, self.simplex), x
                ),
            )
