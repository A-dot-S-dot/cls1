from typing import Optional, Tuple

import defaults
import numpy as np
from benchmark.shallow_water import ShallowWaterBenchmark

import core


def nullify(dof_vector: np.ndarray, eps=defaults.EPSILON) -> np.ndarray:
    height_nullification = np.ones(dof_vector.shape)
    height_nullification = (dof_vector[:, 0] > eps)[:, None]
    discharge_nullification = np.ones(dof_vector.shape)
    discharge_nullification[:, 1] = np.abs(dof_vector[:, 1]) > eps

    return dof_vector * height_nullification * discharge_nullification


def get_average(value_left: np.ndarray, value_right: np.ndarray) -> np.ndarray:
    return np.average((value_left, value_right), axis=0)


def get_height(dof_vector: np.ndarray):
    return dof_vector[:, 0].copy() if len(dof_vector.shape) == 2 else dof_vector[0]


def get_heights(*dof_vector: np.ndarray) -> Tuple:
    return tuple(get_height(dof) for dof in dof_vector)


def get_discharge(dof_vector: np.ndarray):
    return dof_vector[:, 1].copy() if len(dof_vector.shape) == 2 else dof_vector[1]


def get_discharges(*dof_vector: np.ndarray) -> Tuple[np.ndarray, ...]:
    return tuple(get_discharge(dof) for dof in dof_vector)


def get_height_and_discharge(dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return get_height(dof_vector), get_discharge(dof_vector)


def get_velocity(dof_vector: np.ndarray, eps=defaults.EPSILON) -> np.ndarray:
    height = get_height(dof_vector)
    discharge = get_discharge(dof_vector)

    return (2 * height * discharge) / (height**2 + np.maximum(height, eps) ** 2)


def get_velocities(*dof_vector: np.ndarray) -> Tuple[np.ndarray, ...]:
    return tuple(get_velocity(dof) for dof in dof_vector)


def get_height_positivity_fix(
    bar_state: np.ndarray, bathymetry_step: np.ndarray
) -> np.ndarray:
    """See 'Bound-preserving and entropy-stable algebraic flux correction
    schemes for the shallow water equations with topography' by Hajduk and
    Kuzmin.

    """
    return np.minimum(np.abs(bathymetry_step) / 2, bar_state) * np.sign(bathymetry_step)


def is_constant(bathymetry: np.ndarray, eps=defaults.EPSILON) -> bool:
    return (np.abs(bathymetry - bathymetry[0]) < eps).all()


def build_bathymetry_discretization(
    benchmark: ShallowWaterBenchmark, mesh_size: int
) -> np.ndarray:
    mesh = core.UniformMesh(benchmark.domain, mesh_size)
    interpolator = core.CellAverageInterpolator(mesh, 2)
    return interpolator.interpolate(benchmark.bathymetry)


def assert_constant_bathymetry(benchmark: ShallowWaterBenchmark, mesh_size: int):
    bathymetry = build_bathymetry_discretization(benchmark, mesh_size)
    assert is_constant(bathymetry), "Topography is not constant."


class Flux:
    """Returns shallow water flux:

    (q, q**2/h+g*h**2/2)

    """

    _gravitational_acceleration: float
    _eps: float

    def __init__(self, gravitational_acceleration=None, eps=defaults.EPSILON):
        self._gravitational_acceleration = (
            gravitational_acceleration or defaults.GRAVITATIONAL_ACCELERATION
        )
        self._eps = eps

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        height = get_height(dof_vector)
        discharge = get_discharge(dof_vector)

        if isinstance(height, float):
            height = self._adjust_scalar_height(height)
        else:
            self._adjust_height_vector(height)

        flux = np.array(
            [
                discharge,
                discharge**2 / height
                + self._gravitational_acceleration * height**2 / 2,
            ]
        ).T
        flux[np.isnan(height)] = 0.0

        return flux

    def _adjust_scalar_height(self, height: float) -> float:
        if height < 0:
            raise ValueError("Height is negative.")
        elif height < self._eps:
            return np.nan
        else:
            return height

    def _adjust_height_vector(self, height: np.ndarray):
        if (height < 0).any():
            raise ValueError("Height is negative.")

        height[height < self._eps] = np.nan


class WaveSpeed:
    """Calculates local wave velocities for each  Riemann Problem, i.e. it contains

    wave_speed_left = max(0, uL-sqrt(g*hL), uR-sqrt(g*hR))
    wave_speed_right = max(0, uL+sqrt(g*hL), uR+sqrt(g*hR))

    """

    _gravitational_acceleration: float

    def __init__(self, gravitational_acceleration=None):
        self._gravitational_acceleration = (
            gravitational_acceleration or defaults.GRAVITATIONAL_ACCELERATION
        )

    def __call__(self, value_left, value_right) -> Tuple:
        height_left, height_right = get_heights(value_left, value_right)
        velocity_left, velocity_right = get_velocities(value_left, value_right)

        return self._build_wave_speed_left(
            height_left, height_right, velocity_left, velocity_right
        ), self._build_wave_speed_right(
            height_left, height_right, velocity_left, velocity_right
        )

    def _build_wave_speed_left(
        self, height_left, height_right, velocity_left, velocity_right
    ):
        return np.minimum(
            np.minimum(
                velocity_left
                + -np.sqrt(self._gravitational_acceleration * height_left),
                velocity_right
                + -np.sqrt(self._gravitational_acceleration * height_right),
            ),
            0,
        )

    def _build_wave_speed_right(
        self, height_left, height_right, velocity_left, velocity_right
    ):
        return np.maximum(
            np.maximum(
                velocity_left + np.sqrt(self._gravitational_acceleration * height_left),
                velocity_right
                + np.sqrt(self._gravitational_acceleration * height_right),
            ),
            0,
        )


class MaximumWaveSpeed:
    """Calculates local wave velocities for each  Riemann Problem, i.e. it contains

    wave_speed = max(abs(uL)+sqrt(g*hL), |uR| + sqrt(g*hR))

    """

    _gravitational_acceleration: float

    def __init__(self, gravitational_acceleration: float):
        self._gravitational_acceleration = gravitational_acceleration

    def __call__(self, value_left, value_right) -> Tuple:
        height_left, height_right = get_heights(value_left, value_right)
        velocity_left, velocity_right = get_velocities(value_left, value_right)

        wave_speed = self._build_wave_speed(
            height_left, height_right, velocity_left, velocity_right
        )

        return -wave_speed, wave_speed

    def _build_wave_speed(
        self, height_left, height_right, velocity_left, velocity_right
    ):
        return np.maximum(
            np.abs(velocity_left)
            + np.sqrt(self._gravitational_acceleration * height_left),
            np.abs(velocity_right)
            + np.sqrt(self._gravitational_acceleration * height_right),
        )


class RiemannSolver(core.RiemannSolver):
    flux: core.FLUX
    gravitational_acceleration: float

    _scalar_wave_speed: core.WAVE_SPEED

    def __init__(
        self,
        gravitational_acceleration: Optional[float] = None,
        wave_speed=None,
    ):
        self.gravitational_acceleration = (
            gravitational_acceleration or defaults.GRAVITATIONAL_ACCELERATION
        )
        self.flux = Flux(self.gravitational_acceleration)
        self._scalar_wave_speed = wave_speed or MaximumWaveSpeed(
            self.gravitational_acceleration
        )

    def wave_speed(self, value_left: np.ndarray, value_right: np.ndarray) -> Tuple:
        wave_speed_left, wave_speed_right = self._scalar_wave_speed(
            value_left, value_right
        )

        if isinstance(wave_speed_left, float):
            return wave_speed_left, wave_speed_right
        else:
            return wave_speed_left[:, None], wave_speed_right[:, None]

    @property
    def wave_speed_left(self) -> np.ndarray:
        return self._wave_speed_left[:, 0]

    @property
    def wave_speed_right(self) -> np.ndarray:
        return self._wave_speed_right[:, 0]
