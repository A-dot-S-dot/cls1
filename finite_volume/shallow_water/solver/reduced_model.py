import pickle
from typing import Tuple, Type

import defaults
import finite_volume
import finite_volume.shallow_water as swe
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from torch import nn

from .lax_friedrichs import LaxFriedrichsFluxGetter


class ReducedNetwork(NeuralNetRegressor):
    module_type: Type[nn.Module]
    data_path: str
    network_path: str
    optimizer_path: str
    history_path: str
    input_scaler_path: str
    output_scaler_path: str

    _input_scaler: StandardScaler
    _output_scaler: StandardScaler

    def __init__(self, callbacks=None, **kwargs):
        callbacks = callbacks or [EarlyStopping(threshold=1e-8)]
        self._input_scaler = StandardScaler()
        self._output_scaler = StandardScaler()

        super().__init__(
            self.module_type,
            callbacks=callbacks,
            callbacks__print_log__floatfmt=".8f",
            **kwargs,
        )

    def fit(self, X, y, **fit_params):
        X = self._input_scaler.fit_transform(X)
        y = self._output_scaler.fit_transform(y)

        return super().fit(X, y, **fit_params)

    def predict(self, X):
        X = self._input_scaler.transform(X)
        y = super().predict(X)
        return self._output_scaler.inverse_transform(y)

    def save_params(self):
        super().save_params(
            f_params=self.network_path,
            f_optimizer=self.optimizer_path,
            f_history=self.history_path,
        )
        with open(self.input_scaler_path, "wb") as f:
            pickle.dump(self._input_scaler, f)
        with open(self.output_scaler_path, "wb") as f:
            pickle.dump(self._output_scaler, f)

        print(f"Saved network parameters in '{self.network_path}'.")
        print(f"Saved optimizer parameters in '{self.optimizer_path}'.")
        print(f"Saved history in '{self.history_path}'.")
        print(f"Saved input scaler in '{self.input_scaler_path}'.")
        print(f"Saved output scaler in '{self.output_scaler_path}'.")

    def load_params(self):
        self.initialize()
        try:
            super().load_params(
                f_params=self.network_path,
                f_optimizer=self.optimizer_path,
                f_history=self.history_path,
            )
            with open(self.input_scaler_path, "rb") as f:
                self._input_scaler = pickle.load(f)
            with open(self.output_scaler_path, "rb") as f:
                self._output_scaler = pickle.load(f)

        except Exception as error:
            print(f"{error}. Network could not be loaded. Use an untrained network.")


class Curvature:
    step_length: float

    def __init__(self, step_length=2.0):
        self.step_length = step_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        u0 = X[:, :2]
        u1 = X[:, 2:4]
        u2 = X[:, 4:6]
        u3 = X[:, 6:]

        curvature = (
            self._calculate_curvature(u0, u1, u2)
            + self._calculate_curvature(u1, u2, u3)
        ) / 2

        return np.concatenate((X, curvature), axis=1)

    def _calculate_curvature(self, u0, u1, u2):
        return (
            abs(u0 - 2 * u1 + u2)
            * self.step_length
            / (self.step_length**2 + 0.25 * (u0 - u2) ** 2) ** (3 / 2)
        )


class ApproximatedSubgridFlux(finite_volume.NumericalFlux):
    _network: ReducedNetwork

    def __init__(self, input_dimension: int, network: ReducedNetwork):
        self.input_dimension = input_dimension
        self._network = network
        self._network.load_params()

    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        subgrid_flux = self._network.predict(
            np.concatenate(values, axis=1, dtype=np.float32)
        )

        return -subgrid_flux, subgrid_flux


class ReducedFluxGetter(swe.FluxGetter):
    _flux_getter: finite_volume.FluxGetter
    _subgrid_flux: finite_volume.NumericalFlux

    def __init__(
        self,
        input_dimension: int,
        network: ReducedNetwork,
        flux_getter=None,
    ):
        self._flux_getter = flux_getter or LaxFriedrichsFluxGetter()
        self._subgrid_flux = ApproximatedSubgridFlux(input_dimension, network)

    def __call__(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        space: finite_volume.FiniteVolumeSpace,
        bathymetry=None,
    ) -> finite_volume.NumericalFlux:
        bathymetry = bathymetry or swe.build_bathymetry_discretization(
            benchmark, len(space.mesh)
        )
        numerical_flux = self._flux_getter(benchmark, space)
        numerical_flux = finite_volume.CorrectedNumericalFlux(
            numerical_flux, self._subgrid_flux
        )

        return swe.NumericalFlux(
            numerical_flux,
            benchmark.gravitational_acceleration,
            bathymetry=bathymetry,
        )


class ReducedSolverParser(finite_volume.SolverParser):
    _mesh_size_default = defaults.CALCULATE_MESH_SIZE // defaults.COARSENING_DEGREE
    _cfl_default = defaults.FINITE_VOLUME_CFL_NUMBER / defaults.COARSENING_DEGREE

    def _add_arguments(self):
        self._add_ode_solver()
