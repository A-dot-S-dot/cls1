from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .discretization.finite_volume import FiniteVolumeSpace


class NumericalFlux(ABC):
    """Returns left and right flux of each cell."""

    @abstractmethod
    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class NumericalFluxWithHistory(NumericalFlux):
    """Saves numerical flux calculated once in a history."""

    _numerical_flux: NumericalFlux
    _flux_left_history: np.ndarray
    _flux_right_history: np.ndarray

    def __init__(self, numerical_flux: NumericalFlux):
        self._numerical_flux = numerical_flux
        self._flux_left_history = np.array([])
        self._flux_right_history = np.array([])

    @property
    def flux_left_history(self) -> np.ndarray:
        return self._flux_left_history

    @property
    def flux_right_history(self) -> np.ndarray:
        return self._flux_right_history

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flux_left, flux_right = self._numerical_flux(dof_vector)
        self._flux_left_history = np.append(
            self._flux_left_history, np.array([flux_left]), axis=0
        )
        self._flux_right_history = np.append(
            self._flux_right_history, np.array([flux_right]), axis=0
        )

        return flux_left, flux_right


class CorrectedNumericalFlux(NumericalFlux):
    """Adds to a given flux a subgrid flux."""

    _numerical_flux: NumericalFlux
    _flux_correction: NumericalFlux

    def __init__(self, numerical_flux: NumericalFlux, flux_correction: NumericalFlux):
        self._numerical_flux = numerical_flux
        self._flux_correction = flux_correction

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flux_left, flux_right = self._numerical_flux(dof_vector)
        flux_correction_left, flux_correction_right = self._flux_correction(dof_vector)
        return flux_left + flux_correction_left, flux_right + flux_correction_right


class NumericalFluxDependentRightHandSide:
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
