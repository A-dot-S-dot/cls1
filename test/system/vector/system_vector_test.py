from unittest import TestCase

import numpy as np
from system.vector import LocallyAssembledSystemVector, SystemVector
from system.vector.entry_calculator import SystemVectorEntryCalculator

from ...test_helper import LINEAR_LAGRANGE_SPACE


class SimpleVectorEntryCalculator(SystemVectorEntryCalculator):
    def __call__(self, simplex_index: int, local_index: int) -> float:
        return 0


class SimpleSystemVector(SystemVector):
    def assemble(self):
        self[:] = 1


class TestSystemVector(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    vector = SimpleSystemVector(element_space)
    vector.assemble()

    def test_assemble(self):
        self.assertListEqual(
            list(self.vector), list(np.ones(self.element_space.dimension))
        )

    def test_dimension(self):
        self.assertEqual(self.vector.dimension, self.element_space.dimension)

    def test_element_space(self):
        self.assertEqual(self.vector.element_space, self.element_space)


class TestLocallyAssembledSystemVector(TestCase):
    element_space = LINEAR_LAGRANGE_SPACE
    entry_calculator = SimpleVectorEntryCalculator()
    vector = LocallyAssembledSystemVector(element_space, entry_calculator)

    def test_vector(self):
        self.assertListEqual(
            list(self.vector), list(np.zeros(self.element_space.dimension))
        )
