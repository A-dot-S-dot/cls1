import numpy as np
from core.boundary import NodeNeighbours
from core.riemann_solver import RiemannSolver


class OptimalTimeStep:
    _riemann_solver: RiemannSolver
    _node_neighbours: NodeNeighbours
    _step_length: float

    def __init__(
        self,
        riemann_solver: RiemannSolver,
        node_neighbours: NodeNeighbours,
        step_length: float,
    ):
        self._riemann_solver = riemann_solver
        self._node_neighbours = node_neighbours
        self._step_length = step_length

    def __call__(self, time: float, dof_vector: np.ndarray) -> float:
        self._riemann_solver.solve(*self._node_neighbours(dof_vector))

        return self._step_length / np.max(
            np.array(
                [
                    np.abs(self._riemann_solver._wave_speed_left),
                    self._riemann_solver._wave_speed_right,
                ]
            )
        )
