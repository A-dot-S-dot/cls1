from typing import Tuple

import core.ode_solver as os
import defaults
import numpy as np
import shallow_water
import torch
from core import Solver, finite_volume
from core import ode_solver as os
from core import time_stepping as ts
from lib import numerical_flux as nf
from shallow_water.benchmark import ShallowWaterBenchmark
from torch import nn

from . import gmc, godunov, lax_friedrichs


class Curvature:
    step_length: float

    def __init__(self, step_length: float):
        self.step_length = step_length

    def __call__(self, u0, u1, u2, u3):
        return (
            self._calculate_curvature(u0, u1, u2)
            + self._calculate_curvature(u1, u2, u3)
        ) / 2

    def _calculate_curvature(self, u0, u1, u2):
        return (
            abs(u0 - 2 * u1 + u2)
            * self.step_length
            / (self.step_length**2 + 0.25 * (u0 - u2) ** 2) ** (3 / 2)
        )


class Normalization(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean=None, std=None):
        nn.Module.__init__(self)
        self.mean = mean if mean is not None else torch.empty(0)
        self.std = std if std is not None else torch.empty(0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def get_extra_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mean, self.std

    def set_extra_state(self, state):
        self.mean, self.std = state

    def extra_repr(self) -> str:
        return f"mean={self.mean}, std={self.std}"


class NeuralNetwork(nn.Module):
    def __init__(self, mean=None, std=None):
        nn.Module.__init__(self)

        self.normalize = Normalization(mean, std)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        x = self.normalize(x)

        return self.linear_relu_stack(x)


class NetworkSubgridFlux(nf.NumericalFlux):
    """Calculates subgrid flux for shallow water equations with flat bottom. To
    calculate a subgrid flux of a certain edge the Network needs a certain
    number of values of neighboured cells. Requires periodic boundaries."""

    _network: nn.Module
    _local_degree: int
    _curvature: Curvature

    def __init__(
        self,
        volume_space: finite_volume.FiniteVolumeSpace,
        network: nn.Module,
        network_path: str,
        local_degree: int,
    ):
        self._local_degree = local_degree
        self._curvature = Curvature(volume_space.mesh.step_length)
        self._network = network
        self._network.load_state_dict(torch.load(network_path))
        self._network.eval()

    def __call__(self, dof_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input = self._get_input(dof_vector)
        left_subgrid_flux = self._network(input).detach().numpy()

        return left_subgrid_flux, -np.roll(left_subgrid_flux, -1, axis=0)

    def _get_input(self, dof_vector: np.ndarray) -> torch.Tensor:
        stencil_values = np.array(
            [
                np.roll(dof_vector, i, axis=0)
                for i in range(self._local_degree, -self._local_degree, -1)
            ]
        )
        curvature = self._curvature(*stencil_values)

        return torch.Tensor(
            np.concatenate(
                [*stencil_values, curvature],
                axis=1,
            )
        )


class SubgridNetworkSolver(Solver):
    def __init__(
        self,
        benchmark: ShallowWaterBenchmark,
        name=None,
        short=None,
        mesh_size=None,
        coarsening_degree=None,
        local_degree=None,
        cfl_number=None,
        network=None,
        network_path=None,
        save_history=False,
    ):
        benchmark = benchmark
        name = name or "Solver with neural network subgrid flux correction (Godunov)"
        short = short or "subgrid-network"
        coarsening_degree = coarsening_degree or defaults.COARSENING_DEGREE
        mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE // coarsening_degree
        local_degree = local_degree or defaults.LOCAL_DEGREE
        cfl_number = cfl_number or defaults.GODUNOV_CFL_NUMBER / coarsening_degree
        network = network or NeuralNetwork()
        network_path = network_path or defaults.NETWORK_PATH
        ode_solver_type = os.ForwardEuler

        solution = finite_volume.build_finite_volume_solution(
            benchmark, mesh_size, save_history=save_history
        )

        flux = shallow_water.Flux(benchmark.gravitational_acceleration)
        wave_speed = shallow_water.WaveSpeed(
            solution.space, benchmark.gravitational_acceleration
        )
        numerical_flux = godunov.build_godunov_numerical_flux(
            benchmark, solution.space, flux, wave_speed
        )
        subgrid_flux = NetworkSubgridFlux(
            solution.space,
            network,
            network_path,
            local_degree,
        )
        corrected_numerical_flux = nf.CorrectedNumericalFlux(
            numerical_flux, subgrid_flux
        )
        right_hand_side = nf.NumericalFluxDependentRightHandSide(
            solution.space, corrected_numerical_flux
        )

        optimal_time_step = godunov.OptimalTimeStep(
            shallow_water.MaximumWaveSpeed(
                solution.space, benchmark.gravitational_acceleration
            ),
            solution.space.mesh.step_length,
        )
        time_stepping = ts.build_adaptive_time_stepping(
            benchmark, solution, optimal_time_step, cfl_number, adaptive=False
        )
        cfl_checker = ts.CFLChecker(optimal_time_step)

        Solver.__init__(
            self,
            solution,
            right_hand_side,
            ode_solver_type,
            time_stepping,
            name=name,
            short=short,
            cfl_checker=cfl_checker,
        )


class LimitedSubgridNetworkSolver(Solver):
    def __init__(
        self,
        benchmark: ShallowWaterBenchmark,
        name=None,
        short=None,
        mesh_size=None,
        coarsening_degree=None,
        local_degree=None,
        cfl_number=None,
        network=None,
        network_path=None,
        gamma=None,
        save_history=False,
    ):
        benchmark = benchmark
        name = name or "GMC Limited Solver with NN subgrid correction"
        short = short or "limited-subgrid-network"
        coarsening_degree = coarsening_degree or defaults.COARSENING_DEGREE
        mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE // coarsening_degree
        local_degree = local_degree or defaults.LOCAL_DEGREE
        cfl_number = cfl_number or defaults.GODUNOV_CFL_NUMBER / coarsening_degree
        network = network or NeuralNetwork()
        network_path = network_path or defaults.NETWORK_PATH
        gamma = gamma or defaults.GAMMA
        ode_solver_type = os.ForwardEuler

        solution = finite_volume.build_finite_volume_solution(
            benchmark, mesh_size, save_history=save_history
        )

        flux = shallow_water.Flux(benchmark.gravitational_acceleration)
        wave_speed = shallow_water.WaveSpeed(
            solution.space, benchmark.gravitational_acceleration
        )
        numerical_flux = godunov.build_godunov_numerical_flux(
            benchmark, solution.space, flux, wave_speed
        )
        subgrid_flux = NetworkSubgridFlux(
            solution.space,
            network,
            network_path,
            local_degree,
        )
        corrected_flux = nf.CorrectedNumericalFlux(numerical_flux, subgrid_flux)

        wave_speed_max = shallow_water.MaximumWaveSpeed(
            solution.space, benchmark.gravitational_acceleration
        )
        intermediate_state = lax_friedrichs.IntermediateState(
            solution.space, flux, wave_speed_max
        )
        low_order_flux = lax_friedrichs.LLFNumericalFLux(
            solution.space, flux, wave_speed_max, intermediate_state
        )

        local_bounds = gmc.LocalAntidiffusiveFluxBounds(
            solution.space, wave_speed_max, intermediate_state, gamma=gamma
        )
        limited_corrected_flux = gmc.GMCNumericalFlux(
            solution.space, low_order_flux, corrected_flux, local_bounds
        )

        right_hand_side = nf.NumericalFluxDependentRightHandSide(
            solution.space, limited_corrected_flux
        )

        optimal_time_step = godunov.OptimalTimeStep(
            shallow_water.MaximumWaveSpeed(
                solution.space, benchmark.gravitational_acceleration
            ),
            solution.space.mesh.step_length,
        )
        time_stepping = ts.build_adaptive_time_stepping(
            benchmark, solution, optimal_time_step, cfl_number, adaptive=False
        )
        cfl_checker = ts.CFLChecker(optimal_time_step)

        Solver.__init__(
            self,
            solution,
            right_hand_side,
            ode_solver_type,
            time_stepping,
            name=name,
            short=short,
            cfl_checker=cfl_checker,
        )
