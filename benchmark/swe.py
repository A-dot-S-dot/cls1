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
    K1 = 1
    K2 = 25

    name = "Steady State with bump in topography (plot default)"
    short_facts = f"I={domain}, h ca. 2.5, u ca. 0.4, periodic boundaries, T={end_time}, PLOT_DEFAULT"
    description = "This benchmark does not change in time (steady state)."

    def topography(self, x: float) -> float:
        if x in Interval(-0.1, 0.1):
            return (np.cos(10 * np.pi * (x + 1)) + 1) / 4
        else:
            return 0

    def initial_data(self, x: float) -> np.ndarray:
        # calculate root for h after inserting the first mentioned equation in the second one
        f = (
            lambda h: self.K1**2 / (2 * h**2)
            + GRAVITATIONAL_ACCELERATION * (h + self.topography(x))
            - self.K2
        )
        return np.array([newton(f, 2.5), self.K1])

    def exact_solution(self, x: float, t: float) -> np.ndarray:
        return self.initial_data(x)


class SWEOscillationNoTopographyBenchmark(SWEBenchmark):
    """H and u should oscillate about 5% their real height. Initial u and h are
    equal therefore the discharge is h^2.

    """

    domain = Interval(0, LENGTH)
    gravitational_acceleration = GRAVITATIONAL_ACCELERATION
    height_average = HEIGHT_AVERAGE
    height_amplitude = HEIGHT_AMPLITUDE
    height_phase_shift = HEIGHT_PHASE_SHIFT
    height_wave_number = HEIGHT_WAVE_NUMBER
    velocity_average = VELOCITY_AVERAGE
    velocity_amplitude = VELOCITY_AMPLITUDE
    velocity_phase_shift = VELOCITY_PHASE_SHIFT
    velocity_wave_number = VELOCITY_WAVE_NUMBER

    start_time = 0
    end_time = 2 * int(
        domain.length / np.sqrt(gravitational_acceleration * height_average)
    )

    name = "Oscillatory initial data. (animate default)"
    short_facts = f"I={domain}, periodic boundaries, T={end_time}, ANIMATE_DEFAULT "
    description = "Oscillating initial data."

    def topography(self, x: float) -> float:
        return 0

    def initial_data(self, x: float) -> np.ndarray:
        height = self.height_average + self.height_amplitude * np.sin(
            self.height_wave_number * 2 * np.pi * x / self.domain.length
            + self.height_phase_shift
        )
        velocity = self.velocity_average + self.velocity_amplitude * np.sin(
            self.velocity_wave_number * 2 * np.pi * x / self.domain.length
            + self.velocity_phase_shift
        )

        return np.array([height, height * velocity])
