from numpy import cos, exp, pi, sqrt
from pde_solver.mesh import Interval

from .abstract import Benchmark


class AdvectionBenchmark(Benchmark[float]):
    domain = Interval(0, 1)
    start_time = 0
    end_time = 1

    def exact_solution(self, x: float, t: float) -> float:
        argument = (x - t) % self.domain.length
        return self.initial_data(argument)


class AdvectionPlot1Benchmark(AdvectionBenchmark):
    name = "Three Hills (plot default)"
    short_facts = "I=[0,1], periodic boundaries, T=1, PLOT_DEFAULT"
    description = "Three diffrent hills, how the scheme handles different difficulties."

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
    name = "Two Hills"
    short_facts = "I=[0,1], periodic boundaries, T=1"
    description = "Two diffrent hills, how the scheme handles different difficulties."

    def initial_data(self, x: float) -> float:
        if x > 0.5 and x < 0.9:
            return exp(10) * exp(-1 / (x - 0.5)) * exp(1 / (x - 0.9))
        elif x >= 0.2 and x <= 0.4:
            return 1.0
        else:
            return 0.0


class AdvectionPlot3Benchmark(AdvectionBenchmark):
    name = "Moving Rectangle"
    short_facts = "I=[0,1], periodic boundaries, T=1"
    description = "Test for discontinuous initial data."

    def initial_data(self, x: float) -> float:
        if x >= 0.1 and x <= 0.3:
            return 1.0
        else:
            return 0.0


class AdvectionEOCBenchmark1(AdvectionBenchmark):
    name = "Cosine Benchmark (eoc default)"
    short_facts = (
        "u(x)=cos(2*pi*(x-0.5)), I=[0,1], periodic boundaries, T=1, EOC_DEFAULT"
    )
    description = "Smooth initial data for testing convergence order."

    def initial_data(self, x: float) -> float:
        return cos(2 * pi * (x - 0.5))


class AdvectionEOCBenchmark2(AdvectionBenchmark):
    name = "Gaussian Bell"
    short_facts = "u(x)=exp(-100*(x-0.5)^2), I=[0,1], periodic boundaries, T=1"
    description = "Smooth initial data for testing convergence order."

    def initial_data(self, x: float) -> float:
        return exp(-100 * (x - 0.5) ** 2)
