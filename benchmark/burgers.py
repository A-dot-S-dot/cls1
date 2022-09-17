from mesh import Interval
from numpy import pi, sin

from .abstract import Benchmark


class BurgersBenchmark(Benchmark):
    _domain = Interval(0, 1)
    _T = 0.1

    def initial_data(self, x: float) -> float:
        return sin(2 * pi * x)
