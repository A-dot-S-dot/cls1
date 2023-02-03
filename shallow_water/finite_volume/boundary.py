from typing import Literal
import numpy as np
from core import Benchmark, finite_volume

BOUNDARY_CONDITION = finite_volume.BOUNDARY_CONDITION | Literal["wall"]


class ReflectingCondition(finite_volume.BoundaryConditionApplier):
    def add_condition(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        boundary_value = dof_vector[0].copy() if self.left else dof_vector[-1].copy()
        boundary_value[1] = -boundary_value[1]

        if self.left:
            return np.concatenate(
                (
                    np.array([boundary_value for _ in range(self.cells_to_add_number)]),
                    dof_vector,
                ),
                axis=0,
            )
        else:
            return np.concatenate(
                (
                    dof_vector,
                    np.array([boundary_value for _ in range(self.cells_to_add_number)]),
                ),
                axis=0,
            )


class BoundaryConditionsBuilderApplier(finite_volume.BoundaryConditionsApplierBuilder):
    def _build_left_boundary_condition(
        self,
        condition: BOUNDARY_CONDITION,
        benchmark: Benchmark,
        cells_to_add_number: int,
    ) -> finite_volume.BoundaryConditionApplier:
        if condition == "wall":
            return ReflectingCondition("left", cells_to_add_number=cells_to_add_number)
        else:
            return super()._build_left_boundary_condition(
                condition, benchmark, cells_to_add_number
            )

    def _build_right_boundary_condition(
        self,
        condition: BOUNDARY_CONDITION,
        benchmark: Benchmark,
        cells_to_add_number: int,
    ) -> finite_volume.BoundaryConditionApplier:
        if condition == "wall":
            return ReflectingCondition("right", cells_to_add_number=cells_to_add_number)
        else:
            return super()._build_right_boundary_condition(
                condition, benchmark, cells_to_add_number
            )


build_boundary_conditions_applier = BoundaryConditionsBuilderApplier()
