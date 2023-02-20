import numpy as np
from typing import Tuple, overload

from core import CellAverageInterpolator, CustomError, UniformMesh

from .benchmark import ShallowWaterBenchmark
import defaults


class NegativeHeightError(CustomError):
    ...


class Flux:
    """Returns shallow water flux:

    (q, q**2/h+g*h**2/2)

    """

    _gravitational_acceleration: float
    _eps: float

    def __init__(self, gravitational_acceleration: float, eps=defaults.EPSILON):
        self._gravitational_acceleration = gravitational_acceleration
        self._eps = eps

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        height = dof_vector[:, 0].copy()
        discharge = dof_vector[:, 1].copy()

        if (height < 0).any():
            raise NegativeHeightError("Height is negative.")

        height[height < self._eps] = np.nan

        flux = np.array(
            [
                discharge,
                discharge**2 / height
                + self._gravitational_acceleration * height**2 / 2,
            ]
        ).T
        flux[np.isnan(height)] = 0.0

        return flux


def nullify(dof_vector: np.ndarray, eps=defaults.EPSILON) -> np.ndarray:
    return dof_vector * (dof_vector[:, 0] > eps)[:, None]


def get_average(value_left: np.ndarray, value_right: np.ndarray) -> np.ndarray:
    return np.average((value_left, value_right), axis=0)


def get_height(dof_vector: np.ndarray) -> np.ndarray:
    return dof_vector[:, 0]


def get_heights(*dof_vector: np.ndarray) -> Tuple[np.ndarray, ...]:
    return tuple(get_height(dof) for dof in dof_vector)


def get_discharge(dof_vector: np.ndarray) -> np.ndarray:
    return dof_vector[:, 1]


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
    mesh = UniformMesh(benchmark.domain, mesh_size)
    interpolator = CellAverageInterpolator(mesh, 2)
    return interpolator.interpolate(benchmark.bathymetry)


def assert_constant_bathymetry(benchmark: ShallowWaterBenchmark, mesh_size: int):
    bathymetry = build_bathymetry_discretization(benchmark, mesh_size)
    assert is_constant(bathymetry), "Topography is not constant."
