from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from pde_solver.discretization.finite_volume import FiniteVolumeSpace

from .system_vector import SystemVector


class NumericalFlux(ABC):
    """Returns left and right flux of each cell."""

    @abstractmethod
    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class ObservedNumericalFlux(NumericalFlux):
    """Saves numerical flux calculated once."""

    _numerical_flux: NumericalFlux
    _left_numerical_flux: np.ndarray
    _right_numerical_flux: np.ndarray

    def __init__(self, numerical_flux: NumericalFlux):
        self._numerical_flux = numerical_flux

    @property
    def left_numerical_flux(self) -> np.ndarray:
        return self._left_numerical_flux.copy()

    @property
    def right_numerical_flux(self) -> np.ndarray:
        return self._right_numerical_flux.copy()

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._left_numerical_flux, self._right_numerical_flux = self._numerical_flux(
            dof_vector
        )

        return self.left_numerical_flux, self.right_numerical_flux


class NumericalFluxContainer(SystemVector):
    """Saves all numerical fluxes calculated in a row."""

    right_numerical_fluxes: List[np.ndarray]
    left_numerical_fluxes: List[np.ndarray]

    _right_hand_side: SystemVector
    _numerical_flux: ObservedNumericalFlux

    def __init__(
        self, right_hand_side: SystemVector, numerical_flux: ObservedNumericalFlux
    ):
        self._right_hand_side = right_hand_side
        self._numerical_flux = numerical_flux
        self.right_numerical_fluxes, self.left_numerical_fluxes = [], []

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        vector = self._right_hand_side(dof_vector)

        self.right_numerical_fluxes.append(self._numerical_flux.right_numerical_flux)
        self.left_numerical_fluxes.append(self._numerical_flux.left_numerical_flux)

        return vector


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
