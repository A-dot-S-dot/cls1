from typing import Tuple, Union
from unittest import TestCase

import numpy as np
from math_type import MultidimensionalFunction

from ode_solver.explicit_runge_kutta import (
    ForwardEuler,
    Heun,
    RungeKutta8,
    StrongStabilityPreservingRungeKutta3,
    StrongStabilityPreservingRungeKutta4,
)


class TestForwardEuler(TestCase):
    solver_class = ForwardEuler
    target_time = 1.0
    expected_solution_linear = 2.0
    expected_solution_linear_system = np.array([2.0, 4.0])
    accuracy = None

    _start_value: Union[float, np.ndarray]
    _right_hand_side_function: MultidimensionalFunction
    _expected_solution: Union[float, np.ndarray]

    def test_constant_right_side(self):
        """Test with the ode x'(t)=1.

        The exact solution is x(t)=t+x0.

        """
        self._right_hand_side_function = lambda _: np.array(1.0)
        self._start_value = 0.0
        self._expected_solution = self.target_time

        time_steps_numbers = [1, 10, 100, 1000]

        for time_step_number in time_steps_numbers:
            self._test_solution_and_time(time_step_number)

    def _test_solution_and_time(self, time_step_number: int = 1):
        solution_time, solution = self._calculate_ode_solution(time_step_number)

        self.assertEqual(solution_time, self.target_time)
        self._test_solution(solution)

    def _calculate_ode_solution(
        self,
        time_step_number: int,
    ) -> Tuple[float, np.ndarray]:
        solver = self.solver_class()
        solver.right_hand_side_function = self._right_hand_side_function
        solver.set_start_value(np.array(self._start_value).copy())
        time_grid = np.linspace(0, self.target_time, time_step_number + 1)

        for i in range(len(time_grid) - 1):
            delta_t = time_grid[i + 1] - time_grid[i]
            solver.execute_step(delta_t)

        return solver.time, solver.solution

    def _test_solution(self, solution: np.ndarray):
        if isinstance(self._expected_solution, float):
            self.assertAlmostEqual(
                float(solution), self._expected_solution, delta=self.accuracy
            )
        else:
            for solution_i, expected_solution_i in zip(
                solution, self._expected_solution
            ):
                self.assertAlmostEqual(
                    solution_i, expected_solution_i, delta=self.accuracy
                )

    def test_linear_right_side(self):
        """Test with the ode x'(t)=x(t).

        The exact solution is x(t)=x0*exp(t).

        """
        self._right_hand_side_function = lambda x: np.array(x)
        self._start_value = 1.0
        self._expected_solution = self.expected_solution_linear

        self._test_solution_and_time()

    def test_system_constant_right_side(self):
        """Test with the ode (x'(t), y'(t))=(1,1).

        The exact solution is (x(t),y(t))=(t,t)+(x0, y0).

        """
        self._right_hand_side_function = lambda _: np.array([1.0, 1.0])
        self._start_value = np.array([0.0, 0.0])
        self._expected_solution = np.array([self.target_time, self.target_time])

        time_steps_numbers = [1, 10, 100, 1000]

        for time_step_number in time_steps_numbers:
            self._test_solution_and_time(time_step_number)

    def test_system_linear_right_side(self):
        """Test with the ode (x'(t), y'(t))=(x(t), y(t)).

        The exact solution is (x(t), y(t))=(x0*exp(t), y0*exp(t)).

        """
        self._right_hand_side_function = lambda x: np.array(x)
        self._start_value = np.array([1.0, 2.0])
        self._expected_solution = self.expected_solution_linear_system

        self._test_solution_and_time()


class TestHeun(TestForwardEuler):
    solver_class = Heun
    expected_solution_linear = 2.5
    expected_solution_linear_system = np.array([2.5, 5.0])


class TestStrongStabilityPreservingRungeKutta3(TestForwardEuler):
    solver_class = StrongStabilityPreservingRungeKutta3
    expected_solution_linear = 8 / 3
    expected_solution_linear_system = np.array([8 / 3, 16 / 3])


class TestStrongStabilityPreservingRungeKutta4(TestForwardEuler):
    solver_class = StrongStabilityPreservingRungeKutta4
    expected_solution_linear = np.exp(1)
    expected_solution_linear_system = np.array([np.exp(1), 2 * np.exp(1)])
    accuracy = 0.05


class TestRungeKutta8(TestForwardEuler):
    solver_class = RungeKutta8
    expected_solution_linear = np.exp(1)
    expected_solution_linear_system = np.array([np.exp(1), 2 * np.exp(1)])
    accuracy = 1e-4
