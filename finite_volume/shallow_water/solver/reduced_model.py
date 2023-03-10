import pickle
from typing import Dict, Optional, Tuple, Type

import defaults
import finite_volume
import finite_volume.shallow_water as swe
import numpy as np
from torch import nn

from .lax_friedrichs import LaxFriedrichsFluxGetter


class ApproximatedSubgridFlux(finite_volume.NumericalFlux):
    input_dimension = 4
    _network: ...

    def __init__(self, network_path=None):
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

    def __init__(self, flux_getter=None, network_path=None):
        self._network_path = network_path
        self._flux_getter = flux_getter or LaxFriedrichsFluxGetter()

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
        subgrid_flux = ApproximatedSubgridFlux(self._network_path)
        numerical_flux = finite_volume.CorrectedNumericalFlux(
            numerical_flux, subgrid_flux
        )

        return swe.NumericalFlux(
            numerical_flux,
            benchmark.gravitational_acceleration,
            bathymetry=bathymetry,
        )


class ReducedSolver(swe.Solver):
    _network_path: Optional[str]

    def __init__(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        flux_getter=None,
        network_path=None,
        **kwargs,
    ):
        self.flux_getter = ReducedFluxGetter(flux_getter, network_path)
        super().__init__(benchmark, **kwargs)

    def _build_args(
        self,
        benchmark: swe.ShallowWaterBenchmark,
        mesh_size=None,
        cfl_number=None,
        **kwargs,
    ) -> Dict:
        mesh_size = (
            mesh_size or defaults.CALCULATE_MESH_SIZE // defaults.COARSENING_DEGREE
        )
        cfl_number = (
            cfl_number or defaults.FINITE_VOLUME_CFL_NUMBER / defaults.COARSENING_DEGREE
        )
        return super()._build_args(
            benchmark, mesh_size=mesh_size, cfl_number=cfl_number, **kwargs
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
