import numpy as np
from core.boundary import BoundaryConditions
from core.riemann_solver import RiemannSolver


class OptimalTimeStep:
    _riemann_solver: RiemannSolver
    _boundary_conditions: BoundaryConditions
    _step_length: float

    def __init__(
        self,
        riemann_solver: RiemannSolver,
        boundary_conditions: BoundaryConditions,
        step_length: float,
    ):
        self._riemann_solver = riemann_solver
        self._boundary_conditions = boundary_conditions
        self._step_length = step_length

    def __call__(self, time: float, dof_vector: np.ndarray) -> float:
        self._riemann_solver.solve(
            *self._boundary_conditions.get_node_neighbours(dof_vector)
        )

        return self._step_length / np.max(
            np.array(
                [
                    np.abs(self._riemann_solver._wave_speed_left),
                    self._riemann_solver._wave_speed_right,
                ]
            )
        )
