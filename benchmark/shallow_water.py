import random

import numpy as np
from pde_solver.mesh import Interval
from scipy.optimize import newton

from .abstract import Benchmark

GRAVITATIONAL_ACCELERATION = 9.81

# No topography benchmark
LENGTH = 100.0
HEIGHT_AVERAGE = 2.0
HEIGHT_AMPLITUDE = 0.1 * HEIGHT_AVERAGE
HEIGHT_WAVE_NUMBER = 3
HEIGHT_PHASE_SHIFT = 0.0
VELOCITY_AVERAGE = 1.0
VELOCITY_AMPLITUDE = 0.5
VELOCITY_WAVE_NUMBER = 1
VELOCITY_PHASE_SHIFT = np.pi / 2


class ShallowWaterBenchmark(Benchmark[np.ndarray]):
    """All methods returns an array with two quantites. The first one denotes
    the water height. The other one the discharge.

    """

    gravitational_acceleration = GRAVITATIONAL_ACCELERATION

    def topography(self, x: float) -> float:
        raise NotImplementedError


class ShallowWaterBumpSteadyStateBenchmark(ShallowWaterBenchmark):
    """A steady state must fullfill the following equations

        hu=K1,      u^2/2+g(h+b)=K2,

    where K1 and K2 are constants.

    """

    domain = Interval(-2, 2)
    start_time = 0
    end_time: float
    K1: float
    K2: float

    def __init__(self, end_time=None, K1=None, K2=None):
        self.end_time = end_time or 0.1
        self.K1 = K1 or 1.0
        self.K2 = K2 or 25.0

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


class ShallowWaterOscillationNoTopographyBenchmark(ShallowWaterBenchmark):
    """H and u should oscillate about 5% their real height. Initial u and h are
    equal therefore the discharge is h^2.

    """

    domain = Interval(0, LENGTH)
    height_average: float
    height_amplitude: float
    height_phase_shift: float
    height_wave_number: float
    velocity_average: float
    velocity_amplitude: float
    velocity_phase_shift: float
    velocity_wave_number: float

    start_time = 0

    def __init__(
        self,
        end_time=None,
        height_average=None,
        height_amplitude=None,
        height_phase_shift=None,
        height_wave_number=None,
        velocity_average=None,
        velocity_amplitude=None,
        velocity_phase_shift=None,
        velocity_wave_number=None,
    ):
        self.end_time = end_time or 40
        self.height_average = height_average or HEIGHT_AVERAGE
        self.height_amplitude = height_amplitude or HEIGHT_AMPLITUDE
        self.height_phase_shift = height_phase_shift or HEIGHT_PHASE_SHIFT
        self.height_wave_number = height_wave_number or HEIGHT_WAVE_NUMBER
        self.velocity_average = velocity_average or VELOCITY_AVERAGE
        self.velocity_amplitude = velocity_amplitude or VELOCITY_AMPLITUDE
        self.velocity_phase_shift = velocity_phase_shift or VELOCITY_PHASE_SHIFT
        self.velocity_wave_number = velocity_wave_number or VELOCITY_WAVE_NUMBER

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


class ShallowWaterRandomOscillationNoTopographyBenchmark(
    ShallowWaterOscillationNoTopographyBenchmark
):
    domain = Interval(0, LENGTH)
    end_time = 40

    def __init__(
        self,
        end_time=None,
        seed=None,
        height_average=None,
        height_amplitude=None,
        height_phase_shift=None,
        height_wave_number=None,
        velocity_average=None,
        velocity_amplitude=None,
        velocity_phase_shift=None,
        velocity_wave_number=None,
    ):
        random.seed(seed)
        super().__init__(
            end_time=end_time,
            height_average=height_average or random.uniform(1.6, 2.4),
            height_amplitude=height_amplitude or random.uniform(0.2, 0.6),
            height_phase_shift=height_phase_shift or random.uniform(0, 2 * np.pi),
            height_wave_number=height_wave_number or random.randint(1, 5),
            velocity_average=velocity_average or random.uniform(1, 2),
            velocity_amplitude=velocity_amplitude or random.uniform(0.2, 0.6),
            velocity_phase_shift=velocity_phase_shift or random.uniform(0, 2 * np.pi),
            velocity_wave_number=velocity_wave_number or random.randint(1, 4),
        )
