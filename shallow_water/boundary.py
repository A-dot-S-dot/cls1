import numpy as np

import core


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
        *conditions: str,
        radius=1,
        inflow_left=None,
        inflow_right=None,
    ) -> core.BoundaryConditions:
        return super().__call__(
            *conditions,
            radius=radius,
            inflow_left=inflow_left,
            inflow_right=inflow_right,
        )

    def _build_left_cell(
        self, condition: str, index: int, inflow_left=None
    ) -> core.GhostCell:
        if condition == "wall":
            return LeftReflectingCell()
        else:
            return super()._build_left_cell(condition, index, inflow_left=inflow_left)

    def _build_right_cell(
        self, condition: str, index: int, inflow_right=None
    ) -> core.GhostCell:
        if condition == "wall":
            return RightReflectingCell()
        else:
            return super()._build_right_cell(
                condition, index, inflow_right=inflow_right
            )


get_boundary_conditions = BoundaryConditionsBuilder()
