import numpy as np

import core.time_stepping as ts
from core.discretization import DiscreteSolution, FiniteVolumeSpace

from .benchmark import ShallowWaterBenchmark
from .wave_speed import MaximumWaveSpeed


class OptimalTimeStep:
    _wave_speed: MaximumWaveSpeed
    _step_length: float

    def __init__(
        self,
        volume_space: FiniteVolumeSpace,
        gravitational_acceleration: float,
        wave_speed=None,
    ):
        self._wave_speed = wave_speed or MaximumWaveSpeed(
            volume_space, gravitational_acceleration
        )

        self._step_length = volume_space.mesh.step_length

    def __call__(self, dof_vector: np.ndarray) -> float:
        wave_speed = self._wave_speed(dof_vector)
        return self._step_length / (np.max(wave_speed))


def build_adaptive_time_stepping(
    benchmark: ShallowWaterBenchmark,
    solution: DiscreteSolution,
    cfl_number: float,
    adaptive: bool,
) -> ts.TimeStepping:
    return ts.TimeStepping(
        benchmark.end_time,
        cfl_number,
        ts.DiscreteSolutionDependentTimeStep(
            OptimalTimeStep(solution.space, benchmark.gravitational_acceleration),
            solution,
        ),
        adaptive=adaptive,
        start_time=benchmark.start_time,
    )
