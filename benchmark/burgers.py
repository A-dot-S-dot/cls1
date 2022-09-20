from mesh import Interval
from numpy import cos, pi, sin
from scipy.optimize import newton

from .abstract import Benchmark
from warnings import warn


class BurgersBenchmark(Benchmark):
    start_time = 0
    end_time = 0.5

    _critical_time: float
    _warn = True

    def initial_data_derivaitive(self, x: float) -> float:
        raise NotImplementedError

    def exact_solution(self, x: float, t: float) -> float:
        func = lambda u: self.initial_data(x - u * t) - u
        fprime = lambda u: -t * self.initial_data_derivaitive(x - u * t) - 1
        return newton(func, self.initial_data(x), fprime)

    def has_exact_solution(self) -> bool:
        if self.end_time > self._critical_time + 1e-15:
            warn(
                f"End time {self.end_time} after shock formation. No exact solution can be calculated."
            )
            return False
        else:
            return super().has_exact_solution()


class BurgersPlotBenchmark(BurgersBenchmark):
    _domain = Interval(0, 1)
    _critical_time = 1 / (2 * pi)

    def initial_data(self, x: float) -> float:
        return sin(2 * pi * x)

    def initial_data_derivaitive(self, x: float) -> float:
        return 2 * pi * cos(2 * pi * x)


class BurgersEOCBenchmark(BurgersBenchmark):
    _domain = Interval(0, 2 * pi)
    _critical_time = 1

    def initial_data(self, x: float) -> float:
        return sin(x) + 0.5

    def initial_data_derivaitive(self, x: float) -> float:
        return cos(x)
