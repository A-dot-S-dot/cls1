from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .discretization.finite_volume import FiniteVolumeSpace
from .system import SystemVector


class NumericalFlux(ABC):
    """Returns left and right flux of each cell."""

    @abstractmethod
    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class NumericalFluxDependentRightHandSide(SystemVector):
    _volume_space: FiniteVolumeSpace
    _numerical_flux: NumericalFlux

    def __init__(
        self,
        volume_space: FiniteVolumeSpace,
        numerical_flux: NumericalFlux,
    ):
        self._volume_space = volume_space
        self._numerical_flux = numerical_flux

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        left_flux, right_flux = self._numerical_flux(dof_vector)

        step_length = self._volume_space.mesh.step_length
        return (left_flux - right_flux) / step_length
