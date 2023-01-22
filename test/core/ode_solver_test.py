from typing import Generic, Iterable, Tuple, Type, TypeVar
from unittest import TestCase

import numpy as np

import core.ode_solver as os

T = TypeVar("T", float, np.ndarray)


class TestExplicitRungeKutta(TestCase, Generic[T]):
    solver_type: Type[os.ExplicitRungeKuttaMethod]
    target_time = 1.0
    time_step_numbers: Iterable[int] = []
    accuracy = None

    start_value: T
    expected_solution: T

    def test(self):
        for time_step_number in self.time_step_numbers:
            solution_time, solution = self._calculate_ode_solution(time_step_number)

            self.assertEqual(solution_time, self.target_time)
            self._test_solution(solution)

    def _calculate_ode_solution(
        self,
        time_step_number: int,
    ) -> Tuple[float, T]:
        solver = self.solver_type(self.right_hand_side, self.start_value)
        time_grid = np.linspace(0, self.target_time, time_step_number + 1)

        for i in range(len(time_grid) - 1):
            delta_t = time_grid[i + 1] - time_grid[i]
            solver.execute(delta_t)

        return solver.time, solver.solution

    def right_hand_side(self, x: T) -> T:
        ...

    def _test_solution(self, solution: T):
        ...


class TestStageValues(TestCase):
    def test_euler(self):
        solver = os.ForwardEuler(lambda x: 1, 0.0)
        solver.execute(1.0)
        expected_stage_values = [0.0]
        self.assertListEqual(solver.stage_values, expected_stage_values)

    def test_heun(self):
        solver = os.Heun(lambda x: 1, 0.0)
        solver.execute(1.0)
        expected_stage_values = [0.0, 1.0]
        self.assertListEqual(solver.stage_values, expected_stage_values)


class TestConstantEuler(TestExplicitRungeKutta[float]):
    solver_type = os.ForwardEuler
    target_time = 1

    start_value = 0.0
    expected_solution = target_time
    time_step_numbers = [1, 10, 100, 1000]

    def right_hand_side(self, x: float) -> float:
        return 1

    def _test_solution(self, solution: float):
        self.assertAlmostEqual(solution, self.expected_solution, delta=self.accuracy)


class TestLinearEuler(TestConstantEuler):
    start_value = 1
    expected_solution = 2.0
    time_step_numbers = [1]

    def right_hand_side(self, x: float) -> float:
        return x


class TestConstantSystemEuler(TestExplicitRungeKutta[np.ndarray]):
    solver_type = os.ForwardEuler
    target_time = 1

    start_value = np.array([0.0, 0.0])
    expected_solution = np.array([target_time, target_time])
    time_step_numbers = [1, 10, 100, 1000]

    def right_hand_side(self, x: np.ndarray) -> np.ndarray:
        return np.array([1.0, 1.0])

    def _test_solution(self, solution: np.ndarray):
        for solution_i, expected_solution_i in zip(solution, self.expected_solution):
            self.assertAlmostEqual(solution_i, expected_solution_i, delta=self.accuracy)


class TestLinearSystemEuler(TestConstantSystemEuler):
    start_value = np.array([1.0, 2.0])
    expected_solution = np.array([2.0, 4.0])
    time_step_numbers = [1]

    def right_hand_side(self, x: np.ndarray) -> np.ndarray:
        return x


class TestConstantHeun(TestConstantEuler):
    solver_type = os.Heun


class TestLinearHeun(TestLinearEuler):
    solver_type = os.Heun
    expected_solution = 2.5


class TestConstantSystemHeun(TestConstantSystemEuler):
    solver_type = os.Heun


class TestLinearSystemHeun(TestLinearSystemEuler):
    solver_type = os.Heun
    expected_solution = np.array([2.5, 5.0])


class TestConstantStrongStabilityPreservingRungeKutta3(TestConstantEuler):
    solver_type = os.StrongStabilityPreservingRungeKutta3


class TestLinearStrongStabilityPreservingRungeKutta3(TestLinearEuler):
    solver_type = os.StrongStabilityPreservingRungeKutta3

    expected_solution = 8 / 3


class TestConstantSystemStrongStabilityPreservingRungeKutta3(TestConstantSystemEuler):
    solver_type = os.StrongStabilityPreservingRungeKutta3


class TestLinearSystemStrongStabilityPreservingRungeKutta3(TestLinearSystemEuler):
    solver_type = os.StrongStabilityPreservingRungeKutta3
    expected_solution = np.array([8 / 3, 16 / 3])


class TestConstantStrongStabilityPreservingRungeKutta4(TestConstantEuler):
    solver_type = os.StrongStabilityPreservingRungeKutta4


class TestLinearStrongStabilityPreservingRungeKutta4(TestLinearEuler):
    solver_type = os.StrongStabilityPreservingRungeKutta4
    expected_solution = np.exp(1)
    accuracy = 0.05


class TestConstantSystemStrongStabilityPreservingRungeKutta4(TestConstantSystemEuler):
    solver_type = os.StrongStabilityPreservingRungeKutta4


class TestLinearSystemStrongStabilityPreservingRungeKutta4(TestLinearSystemEuler):
    solver_type = os.StrongStabilityPreservingRungeKutta4
    expected_solution = np.array([np.exp(1), 2 * np.exp(1)])
    accuracy = 0.05


class TestConstantRungeKutta8(TestConstantEuler):
    solver_type = os.RungeKutta8


class TestLinearRungeKutta8(TestLinearEuler):
    solver_type = os.RungeKutta8
    expected_solution = np.exp(1)
    accuracy = 1e-4


class TestConstantSystemRungeKutta8(TestConstantSystemEuler):
    solver_type = os.RungeKutta8


class TestLinearSystemRungeKutta8(TestLinearSystemEuler):
    solver_type = os.RungeKutta8
    expected_solution = np.array([np.exp(1), 2 * np.exp(1)])
    accuracy = 1e-4
