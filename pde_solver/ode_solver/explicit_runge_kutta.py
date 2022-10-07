"""This Module contains diffrent methods for Solving ODEs."""
from typing import Callable, Generic, List, TypeVar

import numpy as np
from numpy import sqrt

RealD = TypeVar("RealD", float, np.ndarray)


class ExplicitRungeKuttaMethod(Generic[RealD]):
    """Class for explicit s staged Runge Kutta method for solving the ODE
    x'(t)=f(t,x(t)).
    """

    time: float = 0
    right_hand_side: Callable[[RealD], RealD]

    _runge_kutta_matrix: np.ndarray  # A in Runge-Kutta tableau
    _weights: np.ndarray  # b in Runge-Kutta tableau
    _solution: RealD

    @property
    def start_value(self):
        ...

    @start_value.setter
    def start_value(self, start_value: RealD):
        if isinstance(start_value, np.ndarray):
            self._solution = start_value.copy()
        else:
            self._solution = start_value

    @property
    def solution(self) -> RealD:
        return self._solution

    def execute(self, time_step: float):
        """A Runge Kutta step starting at actual point `xn` at actual time `tn` with the
        time step length `delta_t`.

        """
        stages = []
        stage_number = len(self._weights)

        for index in range(stage_number):
            stage_value = self.solution + time_step * np.sum(
                np.array(
                    [
                        self._runge_kutta_matrix[index, j] * stages[j]
                        for j in range(index)
                    ]
                ),
                axis=0,
            )
            stages.append(self.right_hand_side(stage_value))

        self._solution += time_step * np.sum(
            np.array([weight * stage for weight, stage in zip(self._weights, stages)]),
            axis=0,
        )
        self.time += time_step


class ForwardEuler(ExplicitRungeKuttaMethod):
    """Class for Forward Euler method."""

    _runge_kutta_matrix = np.array([[0]])
    _weights = np.array([1])


class Heun(ExplicitRungeKuttaMethod):
    """Class for Heun's method."""

    _runge_kutta_matrix = np.array([[0, 0], [1, 0]])
    _weights = np.array([1 / 2, 1 / 2])


class StrongStabilityPreservingRungeKutta3(ExplicitRungeKuttaMethod):
    """Class of an optimal third order SSP Runge-Kutta method."""

    _runge_kutta_matrix = np.array([[0, 0, 0], [1, 0, 0], [1 / 4, 1 / 4, 0]])
    _weights = np.array([1 / 6, 1 / 6, 2 / 3])


class StrongStabilityPreservingRungeKutta4(ExplicitRungeKuttaMethod):
    """Class of an optimal fourth order SSP Runge-Kutta method.

    The coefficients are taken from 'Optimal explicit strong stability
    preserving Runge-Kutta methods with high linear order and optimal nonlinear
    order', Gottlieb et al., 2015.

    """

    _coefficients = [
        [0.391752226571890],
        [0.444370493651235, 0.555629506348765, 0.368410593050371],
        [0.620101851488403, 0.379898148511597, 0.251891774271694],
        [0.178079954393132, 0.821920045606868, 0.544974750228521],
        [
            0.517231671970585,
            0.096059710526147,
            0.063692468666290,
            0.386708617503269,
            0.226007483236906,
        ],
    ]
    _stages: List[np.ndarray]

    def execute(self, delta_t: float):
        self._build_empty_stages()
        self._calculate_stages(delta_t)

        self._solution = self._calculate_new_solution(delta_t)
        self.time += delta_t

    def _build_empty_stages(self):
        self._stages = []
        for _ in range(4):
            self._stages.append(np.array(0))

    def _calculate_stages(self, delta_t: float):
        self._calculate_first_stage(delta_t)
        self._calculate_second_stage(delta_t)
        self._calculate_third_stage(delta_t)
        self._calculate_fourth_stage(delta_t)

    def _calculate_first_stage(self, delta_t: float):
        self._stages[0] = self.solution + (
            self._coefficients[0][0] * delta_t * self.right_hand_side(self.solution)
        )

    def _calculate_second_stage(self, delta_t: float):
        self._stages[1] = (
            self._coefficients[1][0] * self.solution
            + self._coefficients[1][1] * self._stages[0]
            + self._coefficients[1][2] * delta_t * self.right_hand_side(self._stages[0])
        )

    def _calculate_third_stage(self, delta_t: float):
        self._stages[2] = (
            self._coefficients[2][0] * self.solution
            + self._coefficients[2][1] * self._stages[1]
            + self._coefficients[2][2] * delta_t * self.right_hand_side(self._stages[1])
        )

    def _calculate_fourth_stage(self, delta_t: float):
        self._stages[3] = (
            self._coefficients[3][0] * self.solution
            + self._coefficients[3][1] * self._stages[2]
            + self._coefficients[3][2] * delta_t * self.right_hand_side(self._stages[2])
        )

    def _calculate_new_solution(self, delta_t: float) -> np.ndarray:
        return (
            self._coefficients[4][0] * self._stages[1]
            + self._coefficients[4][1] * self._stages[2]
            + self._coefficients[4][2] * delta_t * self.right_hand_side(self._stages[2])
            + self._coefficients[4][3] * self._stages[3]
            + self._coefficients[4][4] * delta_t * self.right_hand_side(self._stages[3])
        )


class RungeKutta8(ExplicitRungeKuttaMethod):
    """Class of an eight order Runge-Kutta method.

    Coefficients are taken from "Numerical Methods for Ordinary Differential
    Equations", Second Edition, J.C. Butcher, 2008, ... p. 210, DOI:
    10.1002/9781119121534.ch3

    """

    def __init__(self):
        self._build_runge_kutta_matrix()
        self._build_weights()

    def _build_runge_kutta_matrix(self):
        self._runge_kutta_matrix = np.zeros((11, 11))
        self._build_runge_kutta_matrix_row_2()
        self._build_runge_kutta_matrix_row_3()
        self._build_runge_kutta_matrix_row_4()
        self._build_runge_kutta_matrix_row_5()
        self._build_runge_kutta_matrix_row_6()
        self._build_runge_kutta_matrix_row_7()
        self._build_runge_kutta_matrix_row_8()
        self._build_runge_kutta_matrix_row_9()
        self._build_runge_kutta_matrix_row_10()
        self._build_runge_kutta_matrix_row_11()

    def _build_runge_kutta_matrix_row_2(self):
        self._runge_kutta_matrix[1, 0] = 1 / 2

    def _build_runge_kutta_matrix_row_3(self):
        self._runge_kutta_matrix[2, 0] = 1 / 4
        self._runge_kutta_matrix[2, 1] = 1 / 4

    def _build_runge_kutta_matrix_row_4(self):
        self._runge_kutta_matrix[3, 0] = 1 / 7
        self._runge_kutta_matrix[3, 1] = 1 / 98 * (-7 - 3 * sqrt(21))
        self._runge_kutta_matrix[3, 2] = 1 / 49 * (21 + 5 * sqrt(21))

    def _build_runge_kutta_matrix_row_5(self):
        self._runge_kutta_matrix[4, 0] = 1 / 84 * (11 + sqrt(21))
        self._runge_kutta_matrix[4, 2] = 1 / 63 * (18 + 4 * sqrt(21))
        self._runge_kutta_matrix[4, 3] = 1 / 252 * (21 - sqrt(21))

    def _build_runge_kutta_matrix_row_6(self):
        self._runge_kutta_matrix[5, 0] = 1 / 48 * (5 + sqrt(21))
        self._runge_kutta_matrix[5, 2] = 1 / 36 * (9 + sqrt(21))
        self._runge_kutta_matrix[5, 3] = 1 / 360 * (-231 + 14 * sqrt(21))
        self._runge_kutta_matrix[5, 4] = 1 / 80 * (63 - 7 * sqrt(21))

    def _build_runge_kutta_matrix_row_7(self):
        self._runge_kutta_matrix[6, 0] = 1 / 42 * (10 - sqrt(21))
        self._runge_kutta_matrix[6, 2] = 1 / 315 * (-432 + 92 * sqrt(21))
        self._runge_kutta_matrix[6, 3] = 1 / 90 * (633 - 145 * sqrt(21))
        self._runge_kutta_matrix[6, 4] = 1 / 70 * (-504 + 115 * sqrt(21))
        self._runge_kutta_matrix[6, 5] = 1 / 35 * (63 - 13 * sqrt(21))

    def _build_runge_kutta_matrix_row_8(self):
        self._runge_kutta_matrix[7, 0] = 1 / 14
        self._runge_kutta_matrix[7, 4] = 1 / 126 * (14 - 3 * sqrt(21))
        self._runge_kutta_matrix[7, 5] = 1 / 63 * (13 - 3 * sqrt(21))
        self._runge_kutta_matrix[7, 6] = 1 / 9

    def _build_runge_kutta_matrix_row_9(self):
        self._runge_kutta_matrix[8, 0] = 1 / 32
        self._runge_kutta_matrix[8, 4] = 1 / 576 * (91 - 21 * sqrt(21))
        self._runge_kutta_matrix[8, 5] = 11 / 72
        self._runge_kutta_matrix[8, 6] = 1 / 1152 * (-385 - 75 * sqrt(21))
        self._runge_kutta_matrix[8, 7] = 1 / 128 * (63 + 13 * sqrt(21))

    def _build_runge_kutta_matrix_row_10(self):
        self._runge_kutta_matrix[9, 0] = 1 / 14
        self._runge_kutta_matrix[9, 4] = 1 / 9
        self._runge_kutta_matrix[9, 5] = 1 / 2205 * (-733 - 147 * sqrt(21))
        self._runge_kutta_matrix[9, 6] = 1 / 504 * (515 + 111 * sqrt(21))
        self._runge_kutta_matrix[9, 7] = 1 / 56 * (-51 - 11 * sqrt(21))
        self._runge_kutta_matrix[9, 8] = 1 / 245 * (132 + 28 * sqrt(21))

    def _build_runge_kutta_matrix_row_11(self):
        self._runge_kutta_matrix[10, 4] = 1 / 18 * (-42 + 7 * sqrt(21))
        self._runge_kutta_matrix[10, 5] = 1 / 45 * (-18 + 28 * sqrt(21))
        self._runge_kutta_matrix[10, 6] = 1 / 72 * (-273 - 53 * sqrt(21))
        self._runge_kutta_matrix[10, 7] = 1 / 72 * (301 + 53 * sqrt(21))
        self._runge_kutta_matrix[10, 8] = 1 / 45 * (28 - 28 * sqrt(21))
        self._runge_kutta_matrix[10, 9] = 1 / 18 * (49 - 7 * sqrt(21))

    def _build_weights(self):
        self._weights = np.zeros(11)
        self._weights[0] = 1 / 20
        self._weights[7] = 49 / 180
        self._weights[8] = 16 / 45
        self._weights[9] = 49 / 180
        self._weights[10] = 1 / 20
