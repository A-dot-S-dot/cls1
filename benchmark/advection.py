from mesh import Interval
from numpy import cos, exp, pi, sqrt

from .abstract import Benchmark


class AdvectionBenchmark(Benchmark):
    start_time = 0
    end_time = 1

    _domain = Interval(0, 1)

    def exact_solution(self, x: float, t: float) -> float:
        argument = (x - t) % self.domain.length
        return self.initial_data(argument)


class AdvectionPlot1Benchmark(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        if x >= 0.025 and x < 0.275:
            return exp(-300 * (2 * x - 0.3) ** 2)
        elif x >= 0.35 and x <= 0.55:
            return 1.0
        elif x > 0.7 and x < 0.9:
            return sqrt(1 - ((x - 0.8) / 0.1) ** 2)
        else:
            return 0.0


class AdvectionPlot2Benchmark(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        if x > 0.5 and x < 0.9:
            return exp(10) * exp(-1 / (x - 0.5)) * exp(1 / (x - 0.9))
        elif x >= 0.2 and x <= 0.4:
            return 1.0
        else:
            return 0.0


class AdvectionPlot3Benchmark(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        if x >= 0.1 and x <= 0.3:
            return 1.0
        else:
            return 0.0


class AdvectionEOCBenchmark1(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        return cos(2 * pi * (x - 0.5))


class AdvectionEOCBenchmark2(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        return exp(-100 * (x - 0.5) ** 2)
