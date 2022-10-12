from sys import api_version
import custom_type
import numpy as np
from defaults import *
from pde_solver.mesh import Interval
from scipy.optimize import newton

from .abstract import Benchmark


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

    domain = Interval(-2, 2)
    start_time = 0
    end_time = 0.1
    gravitational_acceleration = 9.81
    K1 = 1
    K2 = 25

    name = "Steady State with bump in topography (plot default)"
    short_facts = f"I={domain}, g={GRAVITATIONAL_ACCELERATION}, h ca. 2.5, u ca. 0.4, periodic boundaries, T={end_time}, PLOT_DEFAULT"
    description = "This benchmark does not change in time (steady state)."

    parser_arguments = {
        "gravitational_acceleration": (
            [
                "+g",
            ],
            {
                "help": "gravitational acceleration",
                "type": custom_type.positive_float,
                "metavar": "ACCELERATION",
                "default": GRAVITATIONAL_ACCELERATION,
                "dest": "gravitational_acceleration",
            },
        )
    }

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


class SWEOscillationNoTopographyBenchmark(SWEBenchmark):
    """H and u should oscillate about 5% their real height. Initial u and h are
    equal therefore the discharge is h^2.

    """

    domain = Interval(0, 2 * np.pi)
    start_time = 0
    end_time = 0.1
    gravitational_acceleration = GRAVITATIONAL_ACCELERATION
    N = 1  # oscillation degree
    base_height = 2.5
    relative_amplitude = 0.05

    name = "Steady State with bump in topography (plot default)"
    short_facts = f"I={domain}, g={GRAVITATIONAL_ACCELERATION}, h ca. 2.5, u ca. 0.4, periodic boundaries, T={end_time}, PLOT_DEFAULT"
    description = "This benchmark does not change in time (steady state)."

    parser_arguments = {
        "N": (
            [
                "+N",
            ],
            {
                "help": "Determines how many oscillations the benchmark should have.",
                "type": custom_type.positive_int,
                "metavar": "OSCILLATION_NUMBER",
                "default": N,
            },
        ),
        "relative_amplitude": (
            ["+a", "++relative-amplitude"],
            {
                "help": "Amplitude relative to the absolute height.",
                "type": custom_type.positive_float,
                "metavar": "PERCENTAGE",
                "default": relative_amplitude,
            },
        ),
    }

    def topography(self, x: float) -> float:
        return 0

    def initial_data(self, x: float) -> np.ndarray:
        h = (
            self.base_height * self.relative_amplitude * np.sin(self.N * x)
            + self.base_height
        )
        return np.array([h, h**2])


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