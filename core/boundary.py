from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Literal, Optional, Tuple, TypeVar

import numpy as np

SIDE = Literal["left", "right"]
T = TypeVar("T", np.ndarray, float)


class GhostCell(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, dof_vector: np.ndarray, time=None) -> T:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class CopyCell(GhostCell, Generic[T]):
    _reference_index: int
    _name: Optional[str]

    def __init__(self, reference_index: int, name=None):
        self._name = name
        self._reference_index = reference_index

    def __call__(self, dof_vector: np.ndarray, time=None) -> T:
        return dof_vector[self._reference_index]

    def __repr__(self) -> str:
        if self._name:
            return self._name
        else:
            return self.__class__.__name__


class InflowCell(GhostCell, Generic[T]):
    inflow: Callable[[float], T]

    def __init__(self, inflow: Callable[[float], T]):
        self.inflow = inflow

    def __call__(self, dof_vector: np.ndarray, time=None) -> T:
        assert time is not None, "TIME must be specified."
        return self.inflow(time)


class BoundaryConditions:
    periodic: bool
    boundary_cells_number: int
    cells_left: List[GhostCell]
    cells_right: List[GhostCell]

    def __init__(self, *cells: GhostCell, periodic=False):
        assert len(cells) % 2 == 0, "An even number of cells is needed."

        self.boundary_cells_number = len(cells) // 2
        self.cells_left = [cell for cell in cells[: self.boundary_cells_number]]
        self.cells_right = [cell for cell in cells[self.boundary_cells_number :]]
        self.periodic = periodic

    def get_node_neighbours(
        self, cell_values: np.ndarray, radius=1, time=None
    ) -> Tuple[np.ndarray, ...]:
        """Returns CELL_VALUES left and right for every node.

        By RADIUS one can specify how many values from left and right
        respectively should be considered. TIME is only used by inflow
        conditions.

        """
        updated_dof_vector = np.array(
            [
                *[cell(cell_values, time=time) for cell in self.cells_left],
                *cell_values,
                *[cell(cell_values, time=time) for cell in self.cells_right],
            ]
        )

        return tuple(
            [
                *[
                    updated_dof_vector[i : i - 2 * radius + 1]
                    for i in range(2 * radius - 1)
                ],
                updated_dof_vector[2 * radius - 1 :],
            ]
        )

    def get_cell_neighbours(
        self, node_values_left: np.ndarray, node_values_right=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns left and right node values for every cell.

        Note, if NODE_VALUES_RIGHT is NONE, NODE_VALUES_LEFT is used.

        """
        if node_values_right is None:
            node_values_right = node_values_left.copy()

        if self.periodic:
            left_value = node_values_right[-2 * self.boundary_cells_number]
            right_value = node_values_left[2 * self.boundary_cells_number - 1]
        else:
            left_value, right_value = np.nan, np.nan

        return np.array([left_value, *node_values_right.copy()]), np.array(
            [*node_values_left.copy(), right_value]
        )

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(cells_left={self.cells_left}, cells_right={self.cells_right}, periodic={self.periodic})"
        )


class BoundaryConditionsBuilder:
    def __call__(
        self,
        *conditions: str,
        radius=1,
        inflow_left=None,
        inflow_right=None,
    ) -> BoundaryConditions:
        condition_left = conditions[0]
        condition_right = conditions[0] if len(conditions) == 1 else conditions[1]
        left_cells = [
            self._build_left_cell(condition_left, i, inflow_left=inflow_left)
            for i in range(radius - 1, -1, -1)
        ]
        right_cells = [
            self._build_right_cell(condition_right, i, inflow_right=inflow_right)
            for i in range(radius)
        ]
        return BoundaryConditions(
            *left_cells, *right_cells, periodic=conditions == ("periodic",)
        )

    def _build_left_cell(
        self, condition: str, index: int, inflow_left=None
    ) -> GhostCell:
        if condition == "periodic":
            return CopyCell(-index - 1, "PeriodicCell")
        elif condition == "outflow":
            return CopyCell(0, "OutflowCell")
        elif condition == "inflow":
            if inflow_left is None:
                raise ValueError("Inflow data required.")
            return InflowCell(inflow_left)
        else:
            raise ValueError(f"No boundary condition can be build for {condition}.")

    def _build_right_cell(
        self, condition: str, index: int, inflow_right=None
    ) -> GhostCell:
        if condition == "periodic":
            return CopyCell(index, "PeriodicCell")
        elif condition == "outflow":
            return CopyCell(-1, "OutflowCell")
        elif condition == "inflow":
            if inflow_right is None:
                raise ValueError("Inflow data required.")
            return InflowCell(inflow_right)
        else:
            raise ValueError(f"No boundary condition can be build for {condition}.")


get_boundary_conditions = BoundaryConditionsBuilder()
