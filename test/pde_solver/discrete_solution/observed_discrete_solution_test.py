from unittest import TestCase

import numpy as np
from pde_solver.discrete_solution import (
    DiscreteSolution,
    DiscreteSolutionObservable,
    DiscreteSolutionObserver,
)


class Observer(DiscreteSolutionObserver):
    updated = False

    def update(self):
        self.updated = True


class TestDiscreteSolution(TestCase):
    initial_data = np.array([0, 0])
    discrete_solution = DiscreteSolution
    observable = DiscreteSolutionObservable(0, initial_data)
    observer = Observer(observable)

    def test_update(self):
        self.observable.add_solution(1, np.array([1, 1]))
        self.assertTrue(self.observer.updated)
