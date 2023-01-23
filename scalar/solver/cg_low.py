import core
import core.ode_solver as os
import defaults
import lib
import numpy as np


class OptimalTimeStep:
    _lumped_mass: core.SystemVector
    _artificial_diffusion: core.SystemMatrix

    def __init__(
        self, lumped_mass: core.SystemVector, artificial_diffusion: core.SystemMatrix
    ):
        self._lumped_mass = lumped_mass
        self._artificial_diffusion = artificial_diffusion

    def __call__(self, dof_vector: np.ndarray) -> float:
        self._artificial_diffusion.assemble(dof_vector)

        return min(
            self._lumped_mass(dof_vector)
            / (2 * abs(self._artificial_diffusion().diagonal()))
        )


class LowOrderCGRightHandSide:
    """Right hand side of low order continuous Galerkin method (ri). To be more
    precise it is defined as following:

        ri = 1/mi*sum(d_ij*(uj-ui)-(fj-fi)*c_ij, j!=i)

    where mi denotes lumped mass, d_ij an artificial diffusion, ui the DOF
    entries, fi flux approximation and c_ij a discrete gradient.

    """

    _lumped_mass: core.SystemVector
    _artificial_diffusion: core.SystemMatrix
    _flux_gradient: core.SystemVector

    def __init__(
        self,
        lumped_mass: core.SystemVector,
        artificial_diffusion: core.SystemMatrix,
        flux_gradient: core.SystemVector,
    ):
        self._lumped_mass = lumped_mass
        self._artificial_diffusion = artificial_diffusion
        self._flux_gradient = flux_gradient

    def __call__(self, dof_vector: np.ndarray) -> np.ndarray:
        self._artificial_diffusion.assemble(dof_vector)
        return (
            self._artificial_diffusion.dot(dof_vector) + self._flux_gradient(dof_vector)
        ) / self._lumped_mass(dof_vector)


def build_cg_low_right_hand_side(
    problem: str, element_space: core.LagrangeSpace
) -> LowOrderCGRightHandSide:
    lumped_mass = lib.LumpedMassVector(element_space)
    artificial_diffusion = lib.build_artificial_diffusion(problem, element_space)
    flux_gradient = lib.build_flux_gradient_approximation(problem, element_space)

    return LowOrderCGRightHandSide(lumped_mass, artificial_diffusion, flux_gradient)


class LowOrderContinuousGalerkinSolver(core.Solver):
    def __init__(
        self,
        benchmark: core.Benchmark,
        name=None,
        short=None,
        mesh_size=None,
        polynomial_degree=None,
        cfl_number=None,
        ode_solver_type=None,
        adaptive=False,
        save_history=False,
    ):
        name = name or "Low order Continuous Galerkin"
        short = short or "cg_low"
        mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE
        polynomial_degree = polynomial_degree or defaults.POLYNOMIAL_DEGREE
        cfl_number = cfl_number or defaults.MCL_CFL_NUMBER
        ode_solver_type = ode_solver_type or os.Heun

        solution = core.build_finite_element_solution(
            benchmark, mesh_size, polynomial_degree, save_history=save_history
        )
        right_hand_side = build_cg_low_right_hand_side(
            benchmark.problem, solution.space
        )
        optimal_time_step = OptimalTimeStep(
            lib.LumpedMassVector(solution.space),
            lib.build_artificial_diffusion(benchmark.problem, solution.space),
        )
        time_stepping = core.build_adaptive_time_stepping(
            benchmark, solution, optimal_time_step, cfl_number, adaptive
        )
        cfl_checker = core.CFLChecker(optimal_time_step)

        core.Solver.__init__(
            self,
            solution,
            right_hand_side,
            ode_solver_type,
            time_stepping,
            name=name,
            short=short,
            cfl_checker=cfl_checker,
        )
