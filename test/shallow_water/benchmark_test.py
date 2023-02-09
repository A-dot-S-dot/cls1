from shallow_water.benchmark import *
from unittest import TestCase


class TestRandomNoTopographyBenchmark(TestCase):
    def test_seed(self):
        benchmark_1 = RandomOscillationNoTopographyBenchmark(seed=0)
        benchmark_2 = RandomOscillationNoTopographyBenchmark(seed=0)

        self.assertDictEqual(vars(benchmark_1), vars(benchmark_2))
