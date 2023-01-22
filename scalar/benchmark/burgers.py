from core.benchmark import Benchmark, NoExactSolutionError
from core.mesh import Interval
from numpy import cos, pi, sin
from scipy.optimize import newton


class BurgersBenchmark(Benchmark[float]):
    problem = "burgers"
    end_time: float

    _critical_time: float
    _warn = True

    def __init__(self, end_time=None):
        self.end_time = end_time or 0.5

    def initial_data_derivaitive(self, x: float) -> float:
        raise NotImplementedError

    def exact_solution(self, x: float, t: float) -> float:
        self._check_shock_formation()

        func = lambda u: self.initial_data(x - u * t) - u
        fprime = lambda u: -t * self.initial_data_derivaitive(x - u * t) - 1
        return newton(func, self.initial_data(x), fprime)

    def _check_shock_formation(self):
        if self.end_time > self._critical_time + 1e-15:
            raise NoExactSolutionError(
                f"End time {self.end_time} is after shock formation. No exact solution can be calculated."
            )


class SchockBenchmark(BurgersBenchmark):
    domain = Interval(0, 1)
    _critical_time = 1 / (2 * pi)

    def initial_data(self, x: float) -> float:
        return sin(2 * pi * x)

    def initial_data_derivaitive(self, x: float) -> float:
        return 2 * pi * cos(2 * pi * x)


class SmoothBenchmark(BurgersBenchmark):
    domain = Interval(0, 2 * pi)
    _critical_time = 1

    def initial_data(self, x: float) -> float:
        return sin(x) + 0.5

    def initial_data_derivaitive(self, x: float) -> float:
        return cos(x)


BENCHMARKS = [
    SchockBenchmark,
    SmoothBenchmark,
]
BENCHMARK_DEFAULTS = {
    "plot": SchockBenchmark,
    "animate": SchockBenchmark,
    "eoc": SmoothBenchmark,
    "calculate": SchockBenchmark,
}
