from typing import List
from unittest import TestCase

import pde_solver.time_stepping as ts


class TestTimeStepFunction:
    t = 0.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


class TestTimeStepGenerator(TestCase):
    def test_constant_time_stepping(self):
        generator = ts.TimeStepGenerator(TestTimeStepFunction(), 0.5)
        self.assertEqual(generator(), 0.5)
        self.assertEqual(generator(), 0.5)

    def test_adaptive_time_stepping(self):
        generator = ts.TimeStepGenerator(TestTimeStepFunction(), 0.5, adaptive=True)
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

    def test_satisfy_cfl_condition(self):
        time_stepping = ts.TimeStepping(1, 1, lambda: 0.5)
        for _ in time_stepping:
            self.assertTrue(time_stepping.satisfy_cfl_condition())
            break

    def test_violate_cfl_condition(self):
        time_stepping = ts.TimeStepping(1, 2, lambda: 0.5)
        for _ in time_stepping:
            self.assertFalse(time_stepping.satisfy_cfl_condition())
            break
