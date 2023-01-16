from typing import List
from unittest import TestCase

import base.time_stepping as ts
import numpy as np


class TestTimeStep:
    t = 0.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


class OptimalTimeStep:
    def __call__(self, dof_vector: np.ndarray) -> float:
        return np.abs(np.max(dof_vector))


class TestCFLChecker(TestCase):
    cfl_checker = ts.CFLChecker(OptimalTimeStep())

    def test_violate_cfl_condition_one_argument(self):
        time_step = 1.0
        dof_vector = np.array([0.5, 0.2])
        self.assertRaises(
            ts.CFLConditionViolatedError, self.cfl_checker, time_step, dof_vector
        )

    def test_violate_cfl_condition_several_arguments(self):
        time_step = 1.0
        dof_vector_1 = np.array([2.0, 3.0])
        dof_vector_2 = np.array([0.5, 0.2])
        self.assertRaises(
            ts.CFLConditionViolatedError,
            self.cfl_checker,
            time_step,
            dof_vector_1,
            dof_vector_2,
        )


class TestTimeStepGenerator(TestCase):
    def test_constant_time_stepping(self):
        generator = ts.TimeStepGenerator(TestTimeStep(), 0.5)
        self.assertEqual(generator(), 0.5)
        self.assertEqual(generator(), 0.5)

    def test_adaptive_time_stepping(self):
        generator = ts.TimeStepGenerator(TestTimeStep(), 0.5, adaptive=True)
        self.assertEqual(generator(), 0.5)
        self.assertEqual(generator(), 1)


class TestTimeStepping(TestCase):
    def test_uniform_time_stepping(self):
        self._test_time_stepping(ts.TimeStepping(1, 1, lambda: 0.5), [0.5, 0.5])

    def _test_time_stepping(
        self, time_stepping: ts.TimeStepping, expected_time_steps: List[float]
    ):
        time_steps = [time_step for time_step in time_stepping]
        self.assertListEqual(time_steps, expected_time_steps)

    def test_not_uniform_time_stepping(self):
        self._test_time_stepping(
            ts.TimeStepping(1, 1, lambda: 0.375), [0.375, 0.375, 0.25]
        )

    def test_to_small_break(self):
        self.assertRaises(
            ts.TimeStepTooSmallError,
            self._test_time_stepping,
            ts.TimeStepping(1, 1, lambda: 1e-13),
            [],
        )

    def test_length(self):
        for generator, expected_length in [(lambda: 0.5, 2), (lambda: 0.375, 3)]:
            time_stepping = ts.TimeStepping(1, 1, generator)
            self.assertEqual(len(time_stepping), expected_length)

    def test_empty_length(self):
        time_stepping = ts.TimeStepping(1, 1, lambda: 1, adaptive=True)
        self.assertEqual(len(time_stepping), 0)
