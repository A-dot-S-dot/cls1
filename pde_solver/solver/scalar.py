import defaults
import pde_solver.ode_solver as os
from benchmark import Benchmark
from pde_solver.discretization.finite_element import LagrangeFiniteElementSpace
from pde_solver.interpolate import Interpolator
from pde_solver.mesh import Mesh
from pde_solver.system_matrix import SystemMatrix
from pde_solver.system_vector import SystemVector

from . import build_functions as bf
from .solver import Solver


class ContinuousGalerkinSolver(Solver):
    mesh_size: int
    polynomial_degree: int
    cfl_number: float
    exact_flux: bool

    mesh: Mesh
    space: LagrangeFiniteElementSpace
    interpolator: Interpolator
    mass: SystemMatrix
    flux_gradient: SystemVector

    def __init__(
        self,
        problem: str,
        benchmark: Benchmark,
        name=None,
        short=None,
        mesh_size=None,
        polynomial_degree=None,
        cfl_number=None,
        exact_flux=False,
    ):
        self.problem = problem
        self.benchmark = benchmark
        self.name = name or "Continuous Galerkin"
        self.short = short or "cg"
        self.mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE
        self.polynomial_degree = polynomial_degree or defaults.POLYNOMIAL_DEGREE
        self.cfl_number = cfl_number or defaults.CFL_NUMBER
        self.exact_flux = exact_flux

        bf.build_uniform_mesh(self)
        bf.build_finite_element_space(self)
        bf.build_optimal_ode_solver(self)
        bf.build_nodal_interpolator(self)
        bf.build_discrete_solution(self)
        bf.build_cg_right_hand_side(self)
        bf.build_mesh_dependent_constant_time_stepping(self)


class LowOrderContinuousGalerkinSolver(Solver):
    mesh_size: int
    polynomial_degree: int
    cfl_number: float
    adaptive: bool

    mesh: Mesh
    space: LagrangeFiniteElementSpace
    interpolator: Interpolator
    lumped_mass: SystemVector
    artificial_diffusion: SystemMatrix
    flux_gradient: SystemVector

    def __init__(
        self,
        problem: str,
        benchmark: Benchmark,
        name=None,
        short=None,
        mesh_size=None,
        polynomial_degree=None,
        cfl_number=None,
        ode_solver=None,
        adaptive=False,
    ):
        self.problem = problem
        self.benchmark = benchmark
        self.name = name or "Low order Continuous Galerkin"
        self.short = short or "cg_low"
        self.mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE
        self.polynomial_degree = polynomial_degree or defaults.POLYNOMIAL_DEGREE
        self.cfl_number = cfl_number or defaults.MCL_CFL_NUMBER
        self.ode_solver = ode_solver or os.Heun()
        self.adaptive = adaptive

        bf.build_uniform_mesh(self)
        bf.build_finite_element_space(self)
        bf.build_nodal_interpolator(self)
        bf.build_discrete_solution(self)
        bf.build_low_cg_right_hand_side(self)
        bf.build_mcl_time_stepping(self)
        bf.build_cfl_checker(self)


class MCLSolver(Solver):
    mesh_size: int
    polynomial_degree: int
    cfl_number: float
    adaptive: bool
    mesh: Mesh

    space: LagrangeFiniteElementSpace
    interpolator: Interpolator
    lumped_mass: SystemVector
    artificial_diffusion: SystemMatrix
    flux_gradient: SystemVector
    flux_approximation: SystemVector

    def __init__(
        self,
        problem: str,
        benchmark: Benchmark,
        name=None,
        short=None,
        mesh_size=None,
        polynomial_degree=None,
        cfl_number=None,
        ode_solver=None,
        adaptive=False,
    ):
        self.problem = problem
        self.benchmark = benchmark
        self.name = name or "MCL Solver"
        self.short = short or "mcl"
        self.mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE
        self.polynomial_degree = polynomial_degree or defaults.POLYNOMIAL_DEGREE
        self.cfl_number = cfl_number or defaults.MCL_CFL_NUMBER
        self.ode_solver = ode_solver or os.Heun()
        self.adaptive = adaptive

        bf.build_uniform_mesh(self)
        bf.build_finite_element_space(self)
        bf.build_nodal_interpolator(self)
        bf.build_discrete_solution(self)
        bf.build_mcl_right_hand_side(self)
        bf.build_mcl_time_stepping(self)
        bf.build_cfl_checker(self)
