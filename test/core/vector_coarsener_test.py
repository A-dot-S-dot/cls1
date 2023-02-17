from unittest import TestCase

import numpy as np

from core import VectorCoarsener


class TestVectorCoarsener(TestCase):
    coarsener = VectorCoarsener(2)

    def test_scalar_coarsening(self):
        vector = np.array([2, 4, 6, 8])
        expected_coarsening = [3, 7]

        self.assertListEqual(list(self.coarsener(vector)), expected_coarsening)

    def test_system_coarsening(self):
        vector = np.array([[2, 4, 6, 8], [0, 2, 4, 6]]).T
        expected_coarsening = [[3, 1], [7, 5]]

        for i in range(2):
            self.assertListEqual(
                list(self.coarsener(vector)[i]), expected_coarsening[i]
            )

    def test_admissible_coarsening_degree(self):
        vector = np.array([1, 2, 3, 4, 5])
        self.assertRaises(ValueError, self.coarsener, vector)
