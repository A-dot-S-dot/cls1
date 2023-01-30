from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
from core import VectorCoarsener
from core.finite_volume import FiniteVolumeSpace


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
    _update_history: Callable[[np.ndarray, np.ndarray], None]

    def __init__(self, numerical_flux: NumericalFlux):
        self._numerical_flux = numerical_flux
        self._flux_left_history = np.empty(0)
        self._flux_right_history = np.empty(0)
        self._update_history = self._initialize_history

    def _initialize_history(self, flux_left: np.ndarray, flux_right: np.ndarray):
        self._flux_left_history = np.array([flux_left])
        self._flux_right_history = np.array([flux_right])

        self._update_history = self._append_to_history

    def _append_to_history(self, flux_left: np.ndarray, flux_right: np.ndarray):
        self._flux_left_history = np.append(
            self._flux_left_history, np.array([flux_left.copy()]), axis=0
        )
        self._flux_right_history = np.append(
            self._flux_right_history, np.array([flux_right.copy()]), axis=0
        )

    @property
    def flux_left_history(self) -> np.ndarray:
        return self._flux_left_history

    @property
    def flux_right_history(self) -> np.ndarray:
        return self._flux_right_history

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flux_left, flux_right = self._numerical_flux(dof_vector)
        self._update_history(flux_left, flux_right)

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


class SubgridFlux(NumericalFlux):
    _fine_flux: NumericalFlux
    _coarse_flux: NumericalFlux
    _coarsener: VectorCoarsener

    def __init__(
        self,
        fine_flux: NumericalFlux,
        coarse_flux: NumericalFlux,
        coarsening_degree: int,
    ):
        self._fine_flux = fine_flux
        self._coarse_flux = coarse_flux
        self._coarsener = VectorCoarsener(coarsening_degree)

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = self._coarsener.coarsening_degree

        fine_flux_left, fine_flux_right = self._fine_flux(dof_vector)

        coarse_dof_vector = self._coarsener(dof_vector)
        coarse_flux_left, coarse_flux_right = self._coarse_flux(coarse_dof_vector)

        subgrid_flux_left = fine_flux_left[::N] + -coarse_flux_left
        subgrid_flux_right = fine_flux_right[N - 1 :: N] + -coarse_flux_right

        return subgrid_flux_left, subgrid_flux_right


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
        return (left_flux + right_flux) / step_length
