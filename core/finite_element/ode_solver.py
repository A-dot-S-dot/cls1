from typing import Type

from core.ode_solver import *

from .space import LagrangeSpace


def build_optimal_ode_solver(
    element_space: LagrangeSpace,
) -> Type[ExplicitRungeKuttaMethod[np.ndarray]]:
    degree = element_space.polynomial_degree
    optimal_solver = {
        1: Heun,
        2: StrongStabilityPreservingRungeKutta3,
        3: RungeKutta8,
        # 3: StrongStabilityPreservingRungeKutta4,
        4: RungeKutta8,
        5: RungeKutta8,
        6: RungeKutta8,
        7: RungeKutta8,
    }

    return optimal_solver[degree]
