from mesh import Interval
from numpy import cos, exp, pi

from .abstract import Benchmark


class AdvectionBenchmark(Benchmark):
    _domain = Interval(0, 1)
    _T = 1

    def exact_solution(self, x: float, t: float) -> float:
        argument = (x - t) % self.domain.length
        return self.initial_data(argument)


class AdvectionPlotBenchmark1(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        return float(x >= 0.2 and x <= 0.4)


class AdvectionEOCBenchmark1(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        return cos(2 * pi * (x - 0.5))


class AdvectionEOCBenchmark2(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        return exp(-100 * (x - 0.5) ** 2)
