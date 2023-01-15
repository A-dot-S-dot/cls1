import pde_solver.interpolate as interpolate
import pde_solver.ode_solver as ode_solver
import pde_solver.system_matrix as matrix
import pde_solver.system_vector as vector
import pde_solver.system_vector.shallow_water_godunov as shallow_water_godunov
import problem.shallow_water as shallow_water
import torch
from pde_solver import network
from pde_solver.discretization import (
    CoarseSolution,
    DiscreteSolution,
    finite_element,
    finite_volume,
)
from pde_solver.mesh import UniformMesh
from pde_solver.time_stepping import TimeStepping


################################################################################
# SOLVER SPACE
################################################################################
def build_uniform_mesh(solver):
    solver.mesh = UniformMesh(solver.benchmark.domain, solver.mesh_size)


def build_finite_element_space(solver):
    solver.space = finite_element.LagrangeFiniteElementSpace(
        solver.mesh, solver.polynomial_degree
    )


def build_finite_volume_space(solver):
    solver.space = finite_volume.FiniteVolumeSpace(solver.mesh)


################################################################################
# INITIAL DATA
################################################################################
def build_nodal_interpolator(solver):
    solver.interpolator = interpolate.NodeValuesInterpolator(*solver.space.basis_nodes)


def build_cell_average_interpolator(solver):
    solver.interpolator = interpolate.CellAverageInterpolator(solver.mesh, 2)


def build_discrete_solution(solver):
    solver.solution = DiscreteSolution(
        solver.interpolator.interpolate(solver.benchmark.initial_data),
        start_time=solver.benchmark.start_time,
        grid=solver.space.grid,
    )


def build_coarse_solution(solver):
    solver.solution = CoarseSolution(
        solver.fine_solver.solution, solver.coarsening_degree
    )


################################################################################
# TIME STEPPING
################################################################################
def build_mesh_dependent_constant_time_stepping(solver):
    solver.time_stepping = TimeStepping(
        solver.benchmark.end_time,
        solver.cfl_number,
        lambda: solver.mesh.step_length,
        start_time=solver.benchmark.start_time,
    )


def build_cfl_checker(solver):
    solver.right_hand_side = vector.CFLCheckedVector(
        solver.right_hand_side, solver.time_stepping
    )


################################################################################
# ODE SOLVER
################################################################################
def build_optimal_ode_solver(solver):
    degree = solver.space.polynomial_degree
    optimal_solver = {
        1: ode_solver.Heun,
        2: ode_solver.StrongStabilityPreservingRungeKutta3,
        3: ode_solver.StrongStabilityPreservingRungeKutta4,
        4: ode_solver.RungeKutta8,
        5: ode_solver.RungeKutta8,
        6: ode_solver.RungeKutta8,
        7: ode_solver.RungeKutta8,
    }

    solver.ode_solver = optimal_solver[degree]()


################################################################################
# CONTINUOUS GALERKIN
################################################################################
def build_mass(solver):
    solver.mass = matrix.MassMatrix(solver.space)


def build_exact_flux_gradient(solver):
    flux_gradients = {
        "advection": vector.AdvectionFluxGradient(solver.space),
        "burgers": vector.FluxGradient(solver.space, lambda u: 1 / 2 * u**2),
    }
    solver.flux_gradient = flux_gradients[solver.problem]


def build_flux_gradient_approximation(solver):
    flux_gradients = {
        "advection": vector.AdvectionFluxGradient(solver.space),
        "burgers": vector.ApproximatedFluxGradient(
            solver.space, lambda u: 1 / 2 * u**2
        ),
    }
    solver.flux_gradient = flux_gradients[solver.problem]


def build_flux_gradient(solver):
    if solver.exact_flux:
        build_exact_flux_gradient(solver)
    else:
        build_flux_gradient_approximation(solver)


def build_cg_right_hand_side(solver):
    build_mass(solver)
    build_flux_gradient(solver)

    solver.right_hand_side = vector.CGRightHandSide(solver.mass, solver.flux_gradient)


################################################################################
# LOW ORDER CONTINUOUS GALERKIN
################################################################################


def build_lumped_mass(solver):
    solver.lumped_mass = vector.LumpedMassVector(solver.space)


def build_artificial_diffusion(solver):
    diffusions = {
        "advection": matrix.DiscreteUpwind,
        "burgers": matrix.BurgersArtificialDiffusion,
    }
    solver.artificial_diffusion = diffusions[solver.problem](solver.space)


def build_low_cg_right_hand_side(solver):
    build_lumped_mass(solver)
    build_flux_gradient_approximation(solver)
    build_artificial_diffusion(solver)

    solver.right_hand_side = vector.LowOrderCGRightHandSide(
        solver.lumped_mass, solver.artificial_diffusion, solver.flux_gradient
    )


################################################################################
# MCL SOLVER
################################################################################


def build_flux_approximation(solver):
    fluxes = {"advection": lambda u: u, "burgers": lambda u: 1 / 2 * u**2}
    solver.flux_approximation = vector.FluxApproximation(fluxes[solver.problem])


def build_mcl_right_hand_side(solver):
    build_low_cg_right_hand_side(solver)
    build_flux_approximation(solver)

    solver.right_hand_side = vector.MCLRightHandSide(
        solver.space, solver.right_hand_side, solver.flux_approximation
    )


def build_mcl_time_stepping(solver):
    solver.time_stepping = TimeStepping(
        solver.benchmark.end_time,
        solver.cfl_number,
        vector.OptimalMCLTimeStep(
            solver.lumped_mass,
            solver.artificial_diffusion,
        ),
        adaptive=solver.adaptive,
        start_time=solver.benchmark.start_time,
    )


################################################################################
# GODUNOV
################################################################################
def build_bottom_topography(solver):
    solver.bottom_topography = solver.interpolator.interpolate(
        solver.benchmark.topography
    )


def build_source_term(solver):
    solver.source_term = shallow_water.NaturalSouceTerm()


def build_godunov_numerical_flux(solver):
    build_bottom_topography(solver)
    build_source_term(solver)

    solver.numerical_flux = shallow_water_godunov.GodunovNumericalFlux(
        solver.space,
        solver.benchmark.gravitational_acceleration,
        solver.bottom_topography,
        source_term=solver.source_term,
    )


def build_godunov_right_hand_side(solver):
    build_godunov_numerical_flux(solver)

    solver.right_hand_side = vector.NumericalFluxDependentRightHandSide(
        solver.space, solver.numerical_flux
    )


def build_shallow_water_godunov_time_stepping(solver):
    solver.time_stepping = TimeStepping(
        solver.benchmark.end_time,
        solver.cfl_number,
        shallow_water_godunov.OptimalTimeStep(
            solver.solution,
            solver.space,
            solver.benchmark.gravitational_acceleration,
            solver.mesh.step_length,
        ),
        adaptive=solver.adaptive,
        start_time=solver.benchmark.start_time,
    )


################################################################################
# EXACT RESOLVED SOLVER (BASED ON GODUNOV)
################################################################################
def build_fine_numerical_flux_container(solver):
    observed_numerical_flux = vector.ObservedNumericalFlux(
        solver.fine_solver.numerical_flux
    )
    solver.fine_numerical_fluxes = vector.NumericalFluxContainer(
        vector.NumericalFluxDependentRightHandSide(
            solver.fine_solver.space, observed_numerical_flux
        ),
        observed_numerical_flux,
    )
    solver.fine_solver.right_hand_side = solver.fine_numerical_fluxes
    solver.fine_solver.solve()


def build_reduced_exact_right_hand_side(solver):
    build_fine_numerical_flux_container(solver)
    build_godunov_numerical_flux(solver)
    solver.numerical_flux = vector.ObservedNumericalFlux(solver.numerical_flux)

    solver.subgrid_flux = vector.ExactSubgridFlux(
        solver.fine_numerical_fluxes,
        solver.solution,
        solver.numerical_flux,
        solver.coarsening_degree,
    )

    solver.right_hand_side = vector.NumericalFluxDependentRightHandSide(
        solver.space,
        vector.CorrectedNumericalFlux(solver.numerical_flux, solver.subgrid_flux),
    )


################################################################################
# RESOLVED SOLVED WHICH SUBGRID FLUXES ARE CALCULATED BY A NETWORK
################################################################################
def build_reduced_network_time_stepping(solver):
    solver.cfl_number = solver.cfl_number / solver.coarsening_degree
    build_shallow_water_godunov_time_stepping(solver)


def setup_network(solver):
    solver.network.load_state_dict(torch.load(solver.network_path))
    solver.network.eval()


def build_reduced_network_right_hand_side(solver):
    build_godunov_numerical_flux(solver)
    curvature = network.Curvature(solver.mesh.step_length)

    solver.subgrid_flux = vector.NetworkSubgridFlux(
        solver.network, curvature, solver.local_degree
    )
    solver.right_hand_side = vector.NumericalFluxDependentRightHandSide(
        solver.space,
        vector.CorrectedNumericalFlux(solver.numerical_flux, solver.subgrid_flux),
    )
