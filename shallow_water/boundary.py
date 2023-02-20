from typing import Literal, Tuple

import numpy as np
import core

BOUNDARY_CONDITION = core.BOUNDARY_CONDITION | Literal["wall"]
BOUNDARY_CONDITIONS = (
    Literal["periodic"] | Tuple[BOUNDARY_CONDITION, BOUNDARY_CONDITION]
)


class LeftReflectingCell(core.GhostCell):
    def __call__(self, dof_vector: np.ndarray, time=None):
        value = dof_vector[0].copy()
        value[1] *= -1

        return value


class RightReflectingCell(core.GhostCell):
    def __call__(self, dof_vector: np.ndarray, time=None):
        value = dof_vector[-1].copy()
        value[1] *= -1

        return value


class BoundaryConditionsBuilder(core.BoundaryConditionsBuilder):
    def __call__(
        self,
        conditions: BOUNDARY_CONDITIONS,
        radius=1,
        inflow_left=None,
        inflow_right=None,
    ) -> core.BoundaryConditions:
        return super().__call__(conditions, radius, inflow_left, inflow_right)

    def _build_left_cell(
        self, condition: BOUNDARY_CONDITION, index: int, inflow_left=None
    ) -> core.GhostCell:
        if condition == "wall":
            return LeftReflectingCell()
        else:
            return super()._build_left_cell(condition, index, inflow_left=inflow_left)

    def _build_right_cell(
        self, condition: BOUNDARY_CONDITION, index: int, inflow_right=None
    ) -> core.GhostCell:
        if condition == "wall":
            return RightReflectingCell()
        else:
            return super()._build_right_cell(
                condition, index, inflow_right=inflow_right
            )


get_boundary_conditions = BoundaryConditionsBuilder()
