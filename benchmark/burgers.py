from mesh import Interval
from numpy import cos, pi, sin
from scipy.optimize import newton

from .abstract import Benchmark


class BurgersBenchmark(Benchmark):
    _domain = Interval(0, 1)
    _T = 0.1

    def initial_data(self, x: float) -> float:
        return sin(2 * pi * x)

    def exact_solution(self, x: float, t: float) -> float:
        func = lambda u: sin(2 * pi * (x - u * t)) - u
        fprime = lambda u: -2 * pi * t * cos(2 * pi * (x - u * t)) - 1

        return newton(func, 0.5 - sin(x), fprime=fprime)
