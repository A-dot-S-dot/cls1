import numpy as np
from pde_solver.mesh import Interval

from .abstract import Benchmark
from scipy.optimize import newton


class SWEBenchmark(Benchmark[np.ndarray]):
    """All methods returns an array with two quantites. The first one denotes
    the water height. The other one the discharge.

    """

    gravitational_acceleration: float

    def topography(self, x: float) -> float:
        raise NotImplementedError


class SWEBumpSteadyStateBenchmark(SWEBenchmark):
    """A steady state must fullfill the following equations

        hu=K1,      u^2/2+g(h+b)=K2,

    where K1 and K2 are constants.

    """

    name = "Steady State with bump in topography (plot default)"
    short_facts = "I=(-2,2), g=9.81, h ca. 2.5, u ca. 0.4, periodic boundaries, T=0.1, PLOT_DEFAULT"
    description = "This benchmark does not change in time (steady state)."

    domain = Interval(-2, 2)
    start_time = 0
    end_time = 0.1
    gravitational_acceleration = 9.81
    K1 = 1
    K2 = 25

    def topography(self, x: float) -> float:
        if x in Interval(-0.1, 0.1):
            return (np.cos(10 * np.pi * (x + 1)) + 1) / 4
        else:
            return 0

    def initial_data(self, x: float) -> np.ndarray:
        # calculate root for h after inserting the first mentioned equation in the second one
        f = (
            lambda h: self.K1**2 / (2 * h**2)
            + self.gravitational_acceleration * (h + self.topography(x))
            - self.K2
        )
        return np.array([newton(f, 2.5), self.K1])

    def exact_solution(self, x: float, t: float) -> np.ndarray:
        return self.initial_data(x)


class SWEWetDryTransitionBenchmark(SWEBenchmark):
    # TODO should als satisfy the above equations
    name = "Wet-Dry transitions"
    short_facts = "I=(-2,2), periodic boundaries, T=0.1"
    description = "This benchmark has wet dry transitions."

    domain = Interval(0, 1)
    start_time = 0
    end_time = 0.1
    gravitational_acceleration = 1

    def topography(self, x: float) -> float:
        if x >= 1 / 3 and x <= 2 / 3:
            return 2
        else:
            return 0

    def initial_data(self, x: float) -> np.ndarray:
        if self.topography(x) > 1:
            height = 0
        else:
            height = 1

        return np.array([height, 0])

    def exact_solution(self, x: float, t: float) -> np.ndarray:
        return self.initial_data(x)
