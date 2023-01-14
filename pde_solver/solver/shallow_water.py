import defaults
import numpy as np
import pde_solver.ode_solver as os
import pde_solver.system_vector as vector
import problem.shallow_water as shallow_water
from benchmark import Benchmark
from pde_solver.discretization.finite_volume import FiniteVolumeSpace
from pde_solver.interpolate import Interpolator
from pde_solver.mesh import Mesh
from pde_solver.network import NeuralNetwork
from torch import nn

from . import build_functions as bf
from .solver import Solver


class GodunovSolver(Solver):
    mesh_size: int
    cfl_number: float
    adaptive: bool

    mesh: Mesh
    space: FiniteVolumeSpace
    interpolator: Interpolator
    bottom_topography: np.ndarray
    intermediate_velocities: shallow_water.WaveSpeed
    numerical_flux: vector.NumericalFlux

    def __init__(
        self,
        problem: str,
        benchmark: Benchmark,
        name=None,
        short=None,
        mesh_size=None,
        cfl_number=None,
        adaptive=False,
    ):
        self.problem = problem
        self.benchmark = benchmark
        self.name = name or "Godunov's finite volume scheme "
        self.short = short or "godunov"
        self.mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE
        self.cfl_number = cfl_number or defaults.GODUNOV_CFL_NUMBER
        self.adaptive = adaptive
        self.ode_solver = os.ForwardEuler()

        bf.build_uniform_mesh(self)
        bf.build_finite_volume_space(self)
        bf.build_cell_average_interpolator(self)
        bf.build_discrete_solution(self)
        bf.build_godunov_right_hand_side(self)
        bf.build_shallow_water_godunov_time_stepping(self)
        bf.build_cfl_checker(self)


class ReducedExactSolver(Solver):
    mesh_size: int
    coarsening_degree: int
    cfl_number: float
    adaptive: bool

    mesh: Mesh
    space: FiniteVolumeSpace
    interpolator: Interpolator
    bottom_topography: np.ndarray
    intermediate_velocities: shallow_water.WaveSpeed
    numerical_flux: vector.NumericalFlux
    fine_solver: Solver
    fine_numerical_fluxes: vector.NumericalFluxContainer
    subgrid_flux: vector.NumericalFlux

    def __init__(
        self,
        problem: str,
        benchmark: Benchmark,
        name=None,
        short=None,
        mesh_size=None,
        coarsening_degree=None,
        cfl_number=None,
        adaptive=False,
    ):
        self.problem = problem
        self.benchmark = benchmark
        self.name = name or "Reduced Exact Solver (Godunov)"
        self.short = short or "reduced-exact"
        self.coarsening_degree = coarsening_degree or defaults.COARSENING_DEGREE
        self.mesh_size = (
            mesh_size or defaults.CALCULATE_MESH_SIZE // self.coarsening_degree
        )
        self.cfl_number = cfl_number or defaults.GODUNOV_CFL_NUMBER
        self.adaptive = adaptive

        self.fine_solver = GodunovSolver(
            problem,
            benchmark,
            name="Generate Fine Solution",
            mesh_size=self.mesh_size * self.coarsening_degree,
            cfl_number=self.cfl_number,
            adaptive=self.adaptive,
        )
        self.ode_solver = self.fine_solver.ode_solver
        self.time_stepping = self.fine_solver.time_stepping

        bf.build_uniform_mesh(self)
        bf.build_finite_volume_space(self)
        bf.build_cell_average_interpolator(self)
        bf.build_coarse_solution(self)
        bf.build_reduced_exact_right_hand_side(self)


class ReducedNetworkSolver(Solver):
    mesh_size: int
    cfl_number: float
    coarsening_degree: int
    local_degree: int
    adaptive = False
    network: nn.Module
    network_path: str

    mesh: Mesh
    space: FiniteVolumeSpace
    interpolator: Interpolator
    bottom_topography: np.ndarray
    intermediate_velocities: shallow_water.WaveSpeed
    numerical_flux: vector.NumericalFlux
    subgrid_flux: vector.NumericalFlux

    def __init__(
        self,
        problem: str,
        benchmark: Benchmark,
        name=None,
        short=None,
        mesh_size=None,
        coarsening_degree=None,
        local_degree=None,
        cfl_number=None,
        network=None,
        network_path=None,
    ):
        self.problem = problem
        self.benchmark = benchmark
        self.name = name or "Reduced Solver with Neural Network (Godunov)"
        self.short = short or "reduced-network"
        self.coarsening_degree = coarsening_degree or defaults.COARSENING_DEGREE
        self.mesh_size = (
            mesh_size or defaults.CALCULATE_MESH_SIZE // self.coarsening_degree
        )
        self.local_degree = local_degree or defaults.LOCAL_DEGREE
        self.cfl_number = cfl_number or defaults.GODUNOV_CFL_NUMBER
        self.network = network or NeuralNetwork()
        self.network_path = network_path or defaults.NETWORK_PATH

        self.ode_solver = os.ForwardEuler()

        bf.build_uniform_mesh(self)
        bf.build_finite_volume_space(self)
        bf.build_cell_average_interpolator(self)
        bf.build_discrete_solution(self)
        bf.setup_network(self)
        bf.build_reduced_network_right_hand_side(self)
        bf.build_reduced_network_time_stepping(self)
