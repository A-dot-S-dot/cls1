import pickle
from typing import Optional, Tuple

import defaults
import finite_volume
import finite_volume.shallow_water as swe
import numpy as np

from .lax_friedrichs import LaxFriedrichsFluxGetter


class Curvature:
    step_length = 2.0

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
    _network: ...

    def __init__(self, input_dimension: int, network_path=None):
        self.input_dimension = input_dimension
        network_path = network_path or defaults.LLF_NETWORK_PATH

        with open(network_path, "rb") as f:
            self._network = pickle.load(f)

    def __call__(self, *values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        subgrid_flux = self._network.predict(
            np.concatenate(values, axis=1, dtype=np.float32)
        )

        return -subgrid_flux, subgrid_flux


class ReducedFluxGetter(swe.FluxGetter):
    _network_path: Optional[str]
    _flux_getter: finite_volume.FluxGetter
    _subgrid_flux: finite_volume.NumericalFlux

    def __init__(self, input_dimension: int, flux_getter=None, network_path=None):
        self._network_path = network_path
        self._flux_getter = flux_getter or LaxFriedrichsFluxGetter()
        self._subgrid_flux = ApproximatedSubgridFlux(input_dimension, network_path)

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
        self._add_network()

    def _add_network(self):
        self.add_argument(
            "++network-file",
            help="Specify file name of the network (file ending is added automatically).",
            metavar="<name>",
            dest="network_file_name",
            default="model",
        )
