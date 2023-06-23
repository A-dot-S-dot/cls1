from typing import Dict

import defaults
import numpy as np
from core import Interval
from core.benchmark import Benchmark
from scipy.optimize import newton

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

    problem = "shallow_water"
    gravitational_acceleration = defaults.GRAVITATIONAL_ACCELERATION

    def bathymetry(self, x: float) -> float:
        raise NotImplementedError


class LakeAtRestNoBathymetryBenchmark(ShallowWaterBenchmark):
    domain = Interval(0, 20)
    _boundary_conditions = "periodic"

    def __init__(self, end_time=100.0):
        self.end_time = end_time

    def bathymetry(self, x: float) -> float:
        return 0

    def initial_data(self, x: float) -> np.ndarray:
        return np.array([1.0, 0])

    def exact_solution(self, x: float, t: float) -> np.ndarray:
        return self.initial_data(x)


class LakeAtRestBenchmark(ShallowWaterBenchmark):
    domain = Interval(0, 20)
    _boundary_conditions = ("outflow", "outflow")

    def __init__(self, end_time=100.0):
        self.end_time = end_time

    def bathymetry(self, x: float) -> float:
        return (4 - (x - 10) ** 2) / 20 if x >= 8.0 and x < 12.0 else 0.0

    def initial_data(self, x: float) -> np.ndarray:
        return np.array([2.0 - self.bathymetry(x), 0])

    def exact_solution(self, x: float, t: float) -> np.ndarray:
        return self.initial_data(x)


class PerturbatedLakeAtRestBenchmark(ShallowWaterBenchmark):
    domain = Interval(0, 20)
    _boundary_conditions = ("outflow", "outflow")

    def __init__(self, end_time=2.0):
        self.end_time = end_time

    def bathymetry(self, x: float) -> float:
        return (4 - (x - 10) ** 2) / 20 if x >= 8.0 and x < 12.0 else 0.0

    def initial_data(self, x: float) -> np.ndarray:
        if x >= 5.75 and x < 6.25:
            return np.array([1.01 - self.bathymetry(x), 0])
        else:
            return np.array([1.0 - self.bathymetry(x), 0])


class MovingWaterNoBathymetryBenchmark(ShallowWaterBenchmark):
    domain = Interval(0, 100)
    _boundary_conditions = "periodic"

    def __init__(self, end_time=1.0):
        self.end_time = end_time

    def bathymetry(self, x: float) -> float:
        return 0

    def initial_data(self, x: float) -> np.ndarray:
        return np.array([2, 1])

    def exact_solution(self, x: float, t: float) -> np.ndarray:
        return self.initial_data(x)


class MovingWaterBumpBathymetryBenchmark(ShallowWaterBenchmark):
    """A steady state must fullfill the following equations

        hu=K1,      u^2/2+g(h+b)=K2,

    where K1 and K2 are constants.

    """

    domain = Interval(-2, 2)
    _boundary_conditions = "periodic"
    K1: float
    K2: float

    def __init__(self, end_time=100.0, K1=1.0, K2=25.0):
        self.end_time = end_time
        self.K1 = K1
        self.K2 = K2

    def bathymetry(self, x: float) -> float:
        if x in Interval(-0.1, 0.1):
            return (np.cos(10 * np.pi * (x + 1)) + 1) / 4
        else:
            return 0

    def initial_data(self, x: float) -> np.ndarray:
        # calculate root for h after inserting the first mentioned equation in the second one
        f = (
            lambda h: self.K1**2 / (2 * h**2)
            + self.gravitational_acceleration * (h + self.bathymetry(x))
            - self.K2
        )
        return np.array([newton(f, 2.5), self.K1])

    def exact_solution(self, x: float, t: float) -> np.ndarray:
        return self.initial_data(x)


class OscillationNoTopographyBenchmark(ShallowWaterBenchmark):
    domain = Interval(0, LENGTH)
    height_average: float
    height_amplitude: float
    height_phase_shift: float
    height_wave_number: float
    velocity_average: float
    velocity_amplitude: float
    velocity_phase_shift: float
    velocity_wave_number: float

    _boundary_conditions = "periodic"

    def __init__(
        self,
        end_time=40.0,
        height_average=None,
        height_amplitude=None,
        height_phase_shift=None,
        height_wave_number=None,
        velocity_average=None,
        velocity_amplitude=None,
        velocity_phase_shift=None,
        velocity_wave_number=None,
    ):
        self.end_time = end_time

        self.height_average = (
            height_average if height_average is not None else HEIGHT_AVERAGE
        )
        self.height_amplitude = (
            height_amplitude if height_amplitude is not None else HEIGHT_AMPLITUDE
        )
        self.height_phase_shift = (
            height_phase_shift if height_phase_shift is not None else HEIGHT_PHASE_SHIFT
        )
        self.height_wave_number = (
            height_wave_number if height_wave_number is not None else HEIGHT_WAVE_NUMBER
        )
        self.velocity_average = (
            velocity_average if velocity_average is not None else VELOCITY_AVERAGE
        )
        self.velocity_amplitude = (
            velocity_amplitude if velocity_amplitude is not None else VELOCITY_AMPLITUDE
        )
        self.velocity_phase_shift = (
            velocity_phase_shift
            if velocity_phase_shift is not None
            else VELOCITY_PHASE_SHIFT
        )
        self.velocity_wave_number = (
            velocity_wave_number
            if velocity_wave_number is not None
            else VELOCITY_WAVE_NUMBER
        )

    def bathymetry(self, x: float) -> float:
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

    def as_dict(self) -> Dict:
        return ShallowWaterBenchmark.as_dict(self) | {
            "h0": self.height_average,
            "Ah": self.height_amplitude,
            "ph": self.height_phase_shift,
            "kh": self.height_wave_number,
            "v0": self.velocity_average,
            "Av": self.velocity_amplitude,
            "pv": self.velocity_phase_shift,
            "kv": self.velocity_wave_number,
        }


class RandomOscillationNoTopographyBenchmark(OscillationNoTopographyBenchmark):
    seed: int
    _generator: np.random.Generator

    def __init__(
        self,
        seed: int,
        end_time=40.0,
        height_average=None,
        height_amplitude=None,
        height_phase_shift=None,
        height_wave_number=None,
        velocity_average=None,
        velocity_amplitude=None,
        velocity_phase_shift=None,
        velocity_wave_number=None,
    ):
        self.seed = seed
        self._generator = np.random.default_rng(seed)

        super().__init__(
            end_time=end_time,
            height_average=height_average or self._generator.uniform(1.6, 2.4),
            height_amplitude=height_amplitude or self._generator.uniform(0.2, 0.6),
            height_phase_shift=height_phase_shift
            or self._generator.uniform(0, 2 * np.pi),
            height_wave_number=height_wave_number or self._generator.integers(1, 6),
            velocity_average=velocity_average or self._generator.uniform(1, 2),
            velocity_amplitude=velocity_amplitude or self._generator.uniform(0.2, 0.6),
            velocity_phase_shift=velocity_phase_shift
            or self._generator.uniform(0, 2 * np.pi),
            velocity_wave_number=velocity_wave_number or self._generator.integers(1, 5),
        )

    def as_dict(self) -> Dict:
        return OscillationNoTopographyBenchmark.as_dict(self) | {"seed": self.seed}


class RandomBenchmarkGenerator:
    _benchmark_kwargs: Dict
    _seed_generator: np.random.Generator

    def __init__(self, seed=None, end_time=None, **kwargs):
        self._seed_generator = np.random.default_rng(seed)
        self._benchmark_kwargs = {"end_time": end_time or 40.0} | kwargs

    def __call__(self, seed=None) -> RandomOscillationNoTopographyBenchmark:
        seed = seed or self._seed_generator.integers(int(1e9))
        return RandomOscillationNoTopographyBenchmark(seed, **self._benchmark_kwargs)

    def __repr__(self) -> str:
        return self.__class__.__name__


class DammBreakBenchmark(ShallowWaterBenchmark):
    h_left: float
    h_right: float
    x_dam: float
    c_mid: float
    _boundary_conditions = ("outflow", "outflow")

    @property
    def g(self) -> float:
        return self.gravitational_acceleration

    def __init__(self, end_time=0.3, h_left=1.0, h_right=0.1, x_dam=0.5, L=1.0, g=1.0):
        assert h_left > h_right, "Left height must be greater than the right one."

        self.domain = Interval(0, L)
        self.end_time = end_time
        self.x_dam = x_dam  # dam location
        self.gravitational_acceleration = g
        self.h_left = h_left
        self.h_right = h_right
        self.c_left = np.sqrt(self.g * self.h_left)
        self.c_right = np.sqrt(self.g * self.h_right)
        self.c_mid = self._get_c_mid()  # wave velocity cm
        self.h_mid = self.c_mid**2 / self.g
        self.u_mid = 2 * (self.c_left - self.c_mid)  # velocity um
        self.v = self.h_mid * self.u_mid / (self.h_mid - self.h_right)  # shock velocity

    def _get_c_mid(self) -> float:
        func = lambda x: (
            x**6
            - 9.0 * self.c_right**2 * x**4
            + 16.0 * self.c_left * self.c_right**2 * x**3
            - self.c_right**2 * (self.c_right**2 + 8.0 * self.c_left**2) * x**2
            + self.c_right**6
        )
        return newton(func, (self.c_left + self.c_right) / 2)

    def bathymetry(self, x: float) -> float:
        return 0

    def initial_data(self, x: float) -> np.ndarray:
        height = self.h_left if x <= self.x_dam else self.h_right

        return np.array([height, 0.0])

    def exact_solution(self, x: float, t: float) -> np.ndarray:
        """See SWASHES by O. Delestre et al."""
        xA = self.x_dam - t * self.c_left
        xB = self.x_dam + t * (2.0 * self.c_left - 3.0 * self.c_mid)
        xC = self.x_dam + self.v * t

        if x <= xA:
            return np.array([self.h_left, 0.0])
        elif x <= xB:
            h = 4 / (9 * self.g) * (self.c_left - (x - self.x_dam) / (2 * t)) ** 2
            u = 2 / 3 * ((x - self.x_dam) / t + self.c_left)
            return np.array([h, h * u])
        elif x <= xC:
            h = self.h_mid
            u = self.u_mid
            return np.array([h, h * u])
        else:
            return np.array([self.h_right, 0.0])


class CylindricalDammBreakWithOutflowBenchmark(ShallowWaterBenchmark):
    domain = Interval(-1, 1)
    gravitational_acceleration = 1.0
    _boundary_conditions = ("outflow", "outflow")

    def __init__(self, end_time=0.2):
        self.end_time = end_time

    def bathymetry(self, x: float) -> float:
        return 0

    def initial_data(self, x: float) -> np.ndarray:
        height = 2.0 if x in Interval(-0.5, 0.5) else 1.0

        return np.array([height, 0.0])


class CylindricalDammBreakWithReflectingBoundaryBenchmark(
    CylindricalDammBreakWithOutflowBenchmark
):
    _boundary_conditions = ("wall", "wall")


class SinusInflowBenchmark(ShallowWaterBenchmark):
    domain = Interval(0, 100)
    _boundary_conditions = ("inflow", "wall")

    def __init__(self, end_time=40):
        self.end_time = end_time

    def inflow_left(self, t: float) -> np.ndarray:
        return np.array([2 + 0.2 * np.sin(t), 0.0])

    def bathymetry(self, x: float) -> float:
        return 0

    def initial_data(self, x: float) -> np.ndarray:
        height = 2.0

        return np.array([height, 0.0])


BENCHMARKS = {
    "lake-at-rest-no-bathymetry": LakeAtRestNoBathymetryBenchmark,
    "lake-at-rest": LakeAtRestBenchmark,
    "lake-at-rest-perturbation": PerturbatedLakeAtRestBenchmark,
    "moving-water-no-bathymetry": MovingWaterNoBathymetryBenchmark,
    "moving-water-bump-bathymetry": MovingWaterBumpBathymetryBenchmark,
    "oscillation": OscillationNoTopographyBenchmark,
    "random": RandomOscillationNoTopographyBenchmark,
    "damm-break": DammBreakBenchmark,
    "cylindrical-damm-break": CylindricalDammBreakWithOutflowBenchmark,
    "cylindrical-damm-break-with-wall": CylindricalDammBreakWithReflectingBoundaryBenchmark,
    "sinus-inflow": SinusInflowBenchmark,
}
BENCHMARK_DEFAULTS = {
    "plot": OscillationNoTopographyBenchmark,
    "animate": OscillationNoTopographyBenchmark,
    "calculate": OscillationNoTopographyBenchmark,
    "plot-error-evolution": OscillationNoTopographyBenchmark,
}
