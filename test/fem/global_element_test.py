from unittest import TestCase

import numpy as np

from fem import GlobalFiniteElement

from ..test_helper import LINEAR_LAGRANGE_SPACE


class TestLagragenFiniteElement(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    element = GlobalFiniteElement(element_space, np.array([2, 2, 2, 2]))
    points = np.array([0.2, 0.4, 0.6, 0.8])

    def test_not_dof_vector_error(self):
        self.assertRaises(ValueError, GlobalFiniteElement, self.element_space, (1, 2))

    def test_element_values(self):
        for point in self.points:
            self.assertEqual(self.element(point), 2)

    def test_element_derivative(self):
        for point in self.points:
            self.assertEqual(self.element.derivative(point), 0)
