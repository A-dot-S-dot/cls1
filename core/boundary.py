from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Literal, Tuple, TypeVar

import numpy as np

SIDE = Literal["left", "right"]
BOUNDARY_CONDITION = Literal["inflow", "outflow", "periodic"]
BOUNDARY_CONDITIONS = (
    Literal["periodic"] | Tuple[BOUNDARY_CONDITION, BOUNDARY_CONDITION]
)
T = TypeVar("T", np.ndarray, float)


class GhostCell(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, dof_vector: np.ndarray, time=None) -> T:
        ...


class CopyCell(GhostCell, Generic[T]):
    _reference_index: int

    def __init__(self, reference_index: int):
        self._reference_index = reference_index

    def __call__(self, dof_vector: np.ndarray, time=None) -> T:
        return dof_vector[self._reference_index]


class InflowCell(GhostCell, Generic[T]):
    inflow: Callable[[float], T]

    def __init__(self, inflow: Callable[[float], T]):
        self.inflow = inflow

    def __call__(self, dof_vector: np.ndarray, time=None) -> T:
        assert time is not None, "TIME must be specified."
        return self.inflow(time)


class NodeNeighbours:
    periodic: bool
    radius: int
    cells_left: List[GhostCell]
    cells_right: List[GhostCell]

    def __init__(self, *cells: GhostCell, radius=None, periodic=False):
        assert len(cells) % 2 == 0, "An even number of cells is needed."

        self.periodic = periodic
        self.radius = radius or len(cells) // 2 or 1
        self.cells_left = [cell for cell in cells[: self.radius]]
        self.cells_right = [cell for cell in cells[self.radius :]]

    def __call__(self, dof_vector: np.ndarray, time=None) -> Tuple[np.ndarray, ...]:
        updated_dof_vector = np.array(
            [
                *[cell(dof_vector, time=time) for cell in self.cells_left],
                *dof_vector,
                *[cell(dof_vector, time=time) for cell in self.cells_right],
            ]
        )

        return tuple(
            [
                *[
                    updated_dof_vector[i : i - 2 * self.radius + 1]
                    for i in range(2 * self.radius - 1)
                ],
                updated_dof_vector[2 * self.radius - 1 :],
            ]
        )


class NodeNeighboursBuilder:
    def __call__(
        self,
        conditions: BOUNDARY_CONDITIONS,
        radius=1,
        inflow_left=None,
        inflow_right=None,
    ) -> NodeNeighbours:
        condition_left = "periodic" if conditions == "periodic" else conditions[0]
        condition_right = "periodic" if conditions == "periodic" else conditions[1]
        left_cells = [
            self._build_left_cell(condition_left, i, inflow_left=inflow_left)
            for i in range(radius - 1, -1, -1)
        ]
        right_cells = [
            self._build_right_cell(condition_right, i, inflow_right=inflow_right)
            for i in range(radius)
        ]
        return NodeNeighbours(
            *left_cells, *right_cells, periodic=conditions == "periodic"
        )

    def _build_left_cell(
        self, condition: BOUNDARY_CONDITION, index: int, inflow_left=None
    ) -> GhostCell:
        if condition == "periodic":
            return CopyCell(-index - 1)
        elif condition == "inflow":
            if inflow_left is None:
                raise ValueError("Inflow data required.")
            return InflowCell(inflow_left)
        else:
            raise ValueError(f"No boundary condition can be build for {condition}.")

    def _build_right_cell(
        self, condition: BOUNDARY_CONDITION, index: int, inflow_right=None
    ) -> GhostCell:
        if condition == "periodic":
            return CopyCell(index)
        elif condition == "outflow":
            return CopyCell(0)
        elif condition == "inflow":
            if inflow_right is None:
                raise ValueError("Inflow data required.")
            return InflowCell(inflow_right)
        else:
            raise ValueError(f"No boundary condition can be build for {condition}.")


build_node_neighbours = NodeNeighboursBuilder()
