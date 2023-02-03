from abc import ABC, abstractmethod
from typing import Callable, Generic, Literal, Tuple, TypeVar

import numpy as np
from core.benchmark import Benchmark

SIDE = Literal["left", "right"]
BOUNDARY_CONDITION = Literal["inflow", "outflow"]
BOUNDARY_CONDITIONS = (
    Literal["periodic"] | Tuple[BOUNDARY_CONDITION, BOUNDARY_CONDITION]
)
T = TypeVar("T", np.ndarray, float)


class BoundaryConditionApplier(ABC):
    _cells_to_add_number: int
    _left: bool
    _right: bool

    def __init__(self, side: SIDE, cells_to_add_number=1):
        self._set_side(side)
        self._cells_to_add_number = cells_to_add_number

    def _set_side(self, side: SIDE):
        if side == "left":
            self._left, self._right = True, False
        elif side == "right":
            self._left, self._right = False, True
        else:
            raise ValueError(f"SIDE {side} is neither 'left' nor 'right'. ")

    @abstractmethod
    def add_condition(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        ...

    @property
    def left(self) -> bool:
        return self._left

    @property
    def right(self) -> bool:
        return self._right

    @property
    def cells_to_add_number(self) -> int:
        return self._cells_to_add_number


class OutflowConditionApplier(BoundaryConditionApplier):
    def add_condition(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        if self.left:
            return np.concatenate(
                (
                    np.array([dof_vector[0] for _ in range(self.cells_to_add_number)]),
                    dof_vector,
                ),
                axis=0,
            )
        else:
            return np.concatenate(
                (
                    dof_vector,
                    np.array([dof_vector[-1] for _ in range(self.cells_to_add_number)]),
                ),
                axis=0,
            )


class InflowConditionApplier(BoundaryConditionApplier, Generic[T]):
    _inflow: Callable[[float], T]

    def __init__(self, side: SIDE, inflow: Callable[[float], T], cells_to_add_number=1):
        BoundaryConditionApplier.__init__(self, side, cells_to_add_number)
        self._inflow = inflow

    def add_condition(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        boundary_value = self._inflow(time)

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


class BoundaryConditionsApplier:
    _condition_applier_left: BoundaryConditionApplier
    _condition_applier_right: BoundaryConditionApplier

    def __init__(
        self,
        condition_applier_left: BoundaryConditionApplier,
        condition_applier_right: BoundaryConditionApplier,
    ):
        self._assert_boundary_side(condition_applier_left, condition_applier_right)

        self._condition_applier_left = condition_applier_left
        self._condition_applier_right = condition_applier_right

    def _assert_boundary_side(
        self,
        condition_applier_left: BoundaryConditionApplier,
        condition_applier_right: BoundaryConditionApplier,
    ):
        if condition_applier_left.right or condition_applier_right.left:
            raise ValueError(
                "BOUNDARY_LEFT or BOUNDARY_RIGHT seem not to be on the right side."
            )

    def add_conditions(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        return self._condition_applier_left.add_condition(
            time, self._condition_applier_right.add_condition(time, dof_vector)
        )

    @property
    def cells_to_add_numbers(self) -> Tuple[int, int]:
        return (
            self._condition_applier_left.cells_to_add_number,
            self._condition_applier_right.cells_to_add_number,
        )


class PeriodicBoundaryConditionsApplier(BoundaryConditionsApplier):
    _cells_to_add_number: Tuple[int, int]

    def __init__(self, cells_to_add_numbers: Tuple[int, int] = (1, 1)):
        self._cells_to_add_numbers = cells_to_add_numbers

    def add_conditions(self, time: float, dof_vector: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                [dof_vector[i] for i in range(-self.cells_to_add_numbers[0], 0)],
                dof_vector,
                [dof_vector[i] for i in range(self.cells_to_add_numbers[1])],
            ),
            axis=0,
        )

    @property
    def cells_to_add_numbers(self) -> Tuple[int, int]:
        return self._cells_to_add_numbers


class BoundaryConditionsApplierBuilder:
    condition = {
        "inflow": InflowConditionApplier,
        "outflow": OutflowConditionApplier,
    }

    def __call__(
        self, benchmark: Benchmark, cells_to_add_numbers: Tuple[int, int] = (1, 1)
    ) -> BoundaryConditionsApplier:
        conditions = benchmark.boundary_conditions

        if conditions == "periodic":
            return PeriodicBoundaryConditionsApplier(
                cells_to_add_numbers=cells_to_add_numbers
            )
        elif len(conditions) == 2:
            condition_left = self._build_left_boundary_condition(
                conditions[0], benchmark, cells_to_add_numbers[0]
            )
            condition_right = self._build_right_boundary_condition(
                conditions[1], benchmark, cells_to_add_numbers[1]
            )
            return BoundaryConditionsApplier(condition_left, condition_right)
        else:
            raise ValueError(f"No boundary conditions can be build for {conditions}.")

    def _build_left_boundary_condition(
        self,
        condition: BOUNDARY_CONDITION,
        benchmark: Benchmark,
        cells_to_add_number: int,
    ) -> BoundaryConditionApplier:
        if condition == "outflow":
            return OutflowConditionApplier(
                "left", cells_to_add_number=cells_to_add_number
            )
        elif condition == "inflow":
            return InflowConditionApplier("left", benchmark.inflow_left)
        else:
            raise ValueError(f"No boundary condition can be build for {condition}.")

    def _build_right_boundary_condition(
        self,
        condition: BOUNDARY_CONDITION,
        benchmark: Benchmark,
        cells_to_add_number: int,
    ) -> BoundaryConditionApplier:
        if condition == "outflow":
            return OutflowConditionApplier(
                "right", cells_to_add_number=cells_to_add_number
            )
        elif condition == "inflow":
            return InflowConditionApplier(
                "right", benchmark.inflow_right, cells_to_add_number=cells_to_add_number
            )
        else:
            raise ValueError(f"No boundary condition can be build for {condition}.")


build_boundary_conditions_applier = BoundaryConditionsApplierBuilder()
