import numpy as np

from core import CellAverageInterpolator, CustomError, UniformMesh

from .benchmark import ShallowWaterBenchmark


class NegativeHeightError(CustomError):
    ...


class Nullifier:
    """Nullifies height and discharges below a certain threshold."""

    _eps: float

    def __init__(self, eps=1e-12):
        self._eps = eps

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        return dof_vector * (dof_vector[:, 0] > self._eps)[:, None]


class DischargeToVelocityTransformer:
    """Returns for a given dof vector which contains heights and discharges a
    dof vector with heights and velocities. To be precise, we obtain (h, q/h).

    """

    _eps: float

    def __init__(self, eps=1e-12):
        self._eps = eps

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        height = dof_vector[:, 0].copy()
        discharge = dof_vector[:, 1].copy()

        if (height < 0).any():
            raise NegativeHeightError

        height[height < self._eps] = np.nan

        transformed_dof_vector = np.array([height, discharge / height]).T
        transformed_dof_vector[np.isnan(height)] = 0.0

        return transformed_dof_vector


class Flux:
    """Returns shallow water flux:

    (q, q**2/h+g*h**2/2)

    """

    _gravitational_acceleration: float
    _eps: float

    def __init__(self, gravitational_acceleration: float, eps=1e-12):
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


def is_constant(bottom: np.ndarray, eps=1e-12) -> bool:
    return (np.abs(bottom - bottom[0]) < eps).all()


def build_topography_discretization(
    benchmark: ShallowWaterBenchmark, mesh_size: int
) -> np.ndarray:
    mesh = UniformMesh(benchmark.domain, mesh_size)
    interpolator = CellAverageInterpolator(mesh, 2)
    return interpolator.interpolate(benchmark.topography)
