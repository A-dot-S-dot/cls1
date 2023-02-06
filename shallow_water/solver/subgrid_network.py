from typing import Tuple

import core.ode_solver as os
import defaults
import lib
import numpy as np
import shallow_water
import torch
from core import Solver, finite_volume
from core import ode_solver as os
from core import time_stepping as ts
from shallow_water.benchmark import ShallowWaterBenchmark
from shallow_water.finite_volume import build_boundary_conditions_applier
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


class NetworkSubgridFlux:
    """Calculates subgrid flux for shallow water equations with flat bottom. To
    calculate a subgrid flux of a certain edge the Network needs a certain
    number of values of neighboured cells. Requires periodic boundaries."""

    _input_radius: int
    _conditions_applier: finite_volume.BoundaryConditionsApplier
    _curvature: Curvature
    _network: nn.Module

    def __init__(
        self,
        input_radius: int,
        conditions_applier: finite_volume.BoundaryConditionsApplier,
        curvature: Curvature,
        network: nn.Module,
        network_path: str,
    ):
        assert conditions_applier.cells_to_add_numbers == (
            input_radius,
            input_radius,
        )

        self._input_radius = input_radius
        self._conditions_applier = conditions_applier
        self._curvature = curvature
        self._network = network

        self._network.load_state_dict(torch.load(network_path))
        self._network.eval()

    def __call__(
        self, time: float, dof_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        dof_vector_with_applied_conditions = self._conditions_applier.add_conditions(
            time, dof_vector
        )
        input = self._get_input(dof_vector_with_applied_conditions)
        subgrid_flux = self._network(input).detach().numpy()

        return subgrid_flux[:-1], -subgrid_flux[1:]

    def _get_input(self, dof_vector: np.ndarray) -> torch.Tensor:
        stencil_values = np.array(
            [np.roll(dof_vector, -i, axis=0) for i in range(2 * self._input_radius)]
        )
        curvature = self._curvature(*stencil_values)
        network_input = np.concatenate(
            [*stencil_values, curvature],
            axis=1,
        )[: -2 * self._input_radius + 1]

        return torch.Tensor(network_input)


class SubgridNetworkSolver(Solver):
    def __init__(
        self,
        benchmark: ShallowWaterBenchmark,
        name=None,
        short=None,
        mesh_size=None,
        coarsening_degree=None,
        input_radius=None,
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
        input_radius = input_radius or defaults.INPUT_RADIUS
        cfl_number = cfl_number or defaults.GODUNOV_CFL_NUMBER / coarsening_degree
        network = network or NeuralNetwork()
        network_path = network_path or defaults.NETWORK_PATH
        ode_solver_type = os.ForwardEuler

        solution = finite_volume.build_finite_volume_solution(
            benchmark,
            mesh_size,
            save_history=save_history,
            periodic=benchmark.boundary_conditions == "periodic",
        )

        numerical_flux = godunov.build_godunov_numerical_flux(
            benchmark, solution.space.mesh
        )

        subgrid_flux = NetworkSubgridFlux(
            input_radius,
            build_boundary_conditions_applier(
                benchmark, cells_to_add_numbers=(input_radius, input_radius)
            ),
            Curvature(solution.space.mesh.step_length),
            network,
            network_path,
        )
        corrected_numerical_flux = lib.CorrectedNumericalFlux(
            numerical_flux, subgrid_flux
        )
        right_hand_side = lib.NumericalFluxDependentRightHandSide(
            solution.space, corrected_numerical_flux
        )

        optimal_time_step = godunov.OptimalTimeStep(
            numerical_flux.riemann_solver, solution.space.mesh.step_length
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
        input_radius=None,
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
        input_radius = input_radius or defaults.INPUT_RADIUS
        cfl_number = cfl_number or defaults.GODUNOV_CFL_NUMBER / coarsening_degree
        network = network or NeuralNetwork()
        network_path = network_path or defaults.NETWORK_PATH
        gamma = gamma or defaults.LIMITING_GAMMA
        ode_solver_type = os.ForwardEuler

        solution = finite_volume.build_finite_volume_solution(
            benchmark, mesh_size, save_history=save_history
        )

        numerical_flux = godunov.build_godunov_numerical_flux(
            benchmark, solution.space.mesh
        )
        subgrid_flux = NetworkSubgridFlux(
            input_radius,
            build_boundary_conditions_applier(
                benchmark, cells_to_add_numbers=(input_radius, input_radius)
            ),
            Curvature(solution.space.mesh.step_length),
            network,
            network_path,
        )

        corrected_flux = lib.CorrectedNumericalFlux(numerical_flux, subgrid_flux)

        limited_corrected_flux = gmc.GMCNumericalFlux(
            solution.space,
            shallow_water.RiemannSolver(
                build_boundary_conditions_applier(benchmark, (1, 1))
            ),
            corrected_flux,
            gamma=gamma,
        )

        right_hand_side = lib.NumericalFluxDependentRightHandSide(
            solution.space, limited_corrected_flux
        )

        optimal_time_step = godunov.OptimalTimeStep(
            limited_corrected_flux.riemann_solver, solution.space.mesh.step_length
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
