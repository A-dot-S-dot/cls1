from itertools import combinations
from typing import Callable

import core
import defaults
import finite_element
import numpy as np

from . import cg_low


class MCLRightHandSide:
    """Right hand side of MCL limiting. To be more
    precise it is defined as following:

        ri = 1/mi*sum(dij*(uj-ui)-(fj-fi)*cij + f*_ij, j!=i)

    where mi denotes lumped mass, dij an artificial diffusion, ui the DOF
    entries, fi flux approximation, cij a discrete gradient and f*_ij a
    corrected flux. The corrected flux is calculated as follows:

        f*_ij = max(fmin_ij, min(fij, fmax_ij)),
        fmax_ij = min(2*dij*umax_i-wij, wji-2*dji*umin_j),
        fmin_ij = max(2*dij*umin_i-wij, wji-2*dji*umax_j),
        wij = dij*(uj+ui)+(fi-fj)*cij,
        fij = mij*(dudt_i-dudt_j)+dij*(ui-uj) (target flux)

    where mij denotes the mass matrix and dudt_i the right hand side of low
    order cg.

    """

    _element_space: finite_element.LagrangeSpace
    _low_cg_right_hand_side: cg_low.LowOrderCGRightHandSide
    _lumped_mass: Callable[[np.ndarray], np.ndarray]
    _artificial_diffusion: core.SystemMatrix
    _flux_approximation: Callable[[np.ndarray], np.ndarray]
    _mass: core.SystemMatrix
    _discrete_gradient: core.SystemMatrix
    _local_maximum: Callable[[np.ndarray], np.ndarray]
    _local_minimum: Callable[[np.ndarray], np.ndarray]

    def __init__(
        self,
        element_space: finite_element.LagrangeSpace,
        low_cg_right_hand_side: cg_low.LowOrderCGRightHandSide,
        flux_approximation: Callable[[np.ndarray], np.ndarray],
    ):
        self._element_space = element_space

        self._low_cg_right_hand_side = low_cg_right_hand_side
        self._lumped_mass = low_cg_right_hand_side._lumped_mass
        self._artificial_diffusion = low_cg_right_hand_side._artificial_diffusion
        self._flux_approximation = flux_approximation
        self._mass = finite_element.MassMatrix(element_space)
        self._discrete_gradient = finite_element.DiscreteGradient(element_space)
        self._local_maximum = core.LocalMaximum(element_space.dof_neighbours)
        self._local_minimum = core.LocalMinimum(element_space.dof_neighbours)

    def __call__(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        corrected_flux = np.zeros(len(dof_vector))
        flux_approximation = self._flux_approximation(dof_vector)
        right_hand_side = self._low_cg_right_hand_side(time, dof_vector)
        local_maximum = self._local_maximum(dof_vector)
        local_minimum = self._local_minimum(dof_vector)

        for element_index in range(len(self._element_space.mesh)):
            for (i_local, j_local) in combinations(
                range(self._element_space.polynomial_degree + 1), 2
            ):
                i = self._element_space.global_index(element_index, i_local)
                j = self._element_space.global_index(element_index, j_local)

                mij = self._mass[i, j]
                dij = self._artificial_diffusion[i, j]
                cij = self._discrete_gradient[i, j]

                ui = dof_vector[i]
                uj = dof_vector[j]
                fi = flux_approximation[i]
                fj = flux_approximation[j]
                dudt_i = right_hand_side[i]
                dudt_j = right_hand_side[j]
                umax_i = local_maximum[i]
                umax_j = local_maximum[j]
                umin_i = local_minimum[i]
                umin_j = local_minimum[j]

                fij = mij * (dudt_i - dudt_j) + dij * (ui - uj)
                wij = dij * (uj + ui) + (fi - fj) * cij

                fmin_ij = max(2 * dij * umin_i - wij, wij - 2 * dij * umax_j)
                fmax_ij = min(2 * dij * umax_i - wij, wij - 2 * dij * umin_j)
                fstar_ij = max(fmin_ij, min(fmax_ij, fij))

                corrected_flux[i] += fstar_ij
                corrected_flux[j] -= fstar_ij

        return right_hand_side + corrected_flux / self._lumped_mass(dof_vector)


def get_mcl_right_hand_side(
    problem: str, element_space: finite_element.LagrangeSpace
) -> MCLRightHandSide:
    flux_approximation = finite_element.build_flux_approximation(problem)

    low_cg_right_hand_side = cg_low.get_cg_low_right_hand_side(problem, element_space)

    return MCLRightHandSide(element_space, low_cg_right_hand_side, flux_approximation)


class MCLSolver(core.Solver):
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
        name = name or "MCL Solver"
        short = short or "mcl"
        mesh_size = mesh_size or defaults.CALCULATE_MESH_SIZE
        polynomial_degree = polynomial_degree or defaults.POLYNOMIAL_DEGREE
        cfl_number = cfl_number or defaults.MCL_CFL_NUMBER
        ode_solver_type = ode_solver_type or core.Heun
        adaptive = adaptive
        solution = finite_element.get_finite_element_solution(
            benchmark, mesh_size, polynomial_degree, save_history=save_history
        )
        right_hand_side = get_mcl_right_hand_side(benchmark.problem, solution.space)
        optimal_time_step = cg_low.OptimalTimeStep(
            finite_element.LumpedMassVector(solution.space),
            finite_element.build_artificial_diffusion(
                benchmark.problem, solution.space
            ),
        )
        time_stepping = core.get_adaptive_time_stepping(
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


class MCLParser(finite_element.SolverParser):
    prog = "mcl-fem"
    name = "Finite Element MCL Solver"
    solver = MCLSolver

    def _add_arguments(self):
        super()._add_arguments(cfl_default=1.0)
        self._add_adaptive_time_stepping()
        self._add_ode_solver()
