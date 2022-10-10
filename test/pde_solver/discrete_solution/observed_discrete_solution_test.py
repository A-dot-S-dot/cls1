from unittest import TestCase

import numpy as np

from pde_solver.discrete_solution import (
    DiscreteSolutionObserver,
    DiscreteSolutionObservable,
)


class Observer(DiscreteSolutionObserver):
    updated = False

    def update(self):
        self.updated = True


class TestDiscreteSolution(TestCase):
    initial_data = np.array([0, 0])
    observable = DiscreteSolutionObservable(0, initial_data)
    observer = Observer()
    observable.register_observer(observer)

    def test_update(self):
        self.observable.add_solution(1, np.array([1, 1]))
        self.assertTrue(self.observer.updated)