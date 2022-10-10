from numpy import cos, pi, sin
from pde_solver.mesh import Interval
from scipy.optimize import newton

from .abstract import Benchmark, NoExactSolutionError


class BurgersBenchmark(Benchmark[float]):
    start_time = 0
    end_time = 0.5

    _critical_time: float
    _warn = True

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


class BurgersPlotBenchmark(BurgersBenchmark):
    name = "Shock formation Benchmark"
    short_facts = "u(x)=sin(2*pi*x), I=[0,1], periodic boundaries, T=0.5, PLOT_DEFAULT"
    description = "A shock occurs after 1/2pi (nach ca. 0.159)). Benchmark is designed to test how a scheme handles it."

    domain = Interval(0, 1)
    _critical_time = 1 / (2 * pi)

    def initial_data(self, x: float) -> float:
        return sin(2 * pi * x)

    def initial_data_derivaitive(self, x: float) -> float:
        return 2 * pi * cos(2 * pi * x)


class BurgersEOCBenchmark(BurgersBenchmark):
    name = "Convergence Order Benchmark"
    short_facts = "u(x)=sin(x)+0.5, I=[0,1], periodic boundaries, T=0.5, EOC_DEFAULT"
    description = "A similar benchmark to the PLOT_DEFAULT one. The shock occurs later such that the solution is smooth and can be used for calculation of order of convergence."

    domain = Interval(0, 2 * pi)
    _critical_time = 1

    def initial_data(self, x: float) -> float:
        return sin(x) + 0.5

    def initial_data_derivaitive(self, x: float) -> float:
        return cos(x)
