from numpy import cos, exp, pi, sqrt
from pde_solver.mesh import Interval

from .abstract import Benchmark


class AdvectionBenchmark(Benchmark[float]):
    domain = Interval(0, 1)
    start_time = 0
    end_time: float

    def __init__(self, end_time=None):
        self.end_time = end_time or 1.0

    def exact_solution(self, x: float, t: float) -> float:
        argument = (x - t) % self.domain.length
        return self.initial_data(argument)


class AdvectionThreeHillsBenchmark(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        if x >= 0.025 and x < 0.275:
            return exp(-300 * (2 * x - 0.3) ** 2)
        elif x >= 0.35 and x <= 0.55:
            return 1.0
        elif x > 0.7 and x < 0.9:
            return sqrt(1 - ((x - 0.8) / 0.1) ** 2)
        else:
            return 0.0


class AdvectionTwoHillsBenchmark(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        if x > 0.5 and x < 0.9:
            return exp(10) * exp(-1 / (x - 0.5)) * exp(1 / (x - 0.9))
        elif x >= 0.2 and x <= 0.4:
            return 1.0
        else:
            return 0.0


class AdvectionOneHillBenchmark(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        if x >= 0.1 and x <= 0.3:
            return 1.0
        else:
            return 0.0


class AdvectionCosineBenchmark(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        return cos(2 * pi * (x - 0.5))


class AdvectionGaussianBellBenchmark(AdvectionBenchmark):
    def initial_data(self, x: float) -> float:
        return exp(-100 * (x - 0.5) ** 2)
