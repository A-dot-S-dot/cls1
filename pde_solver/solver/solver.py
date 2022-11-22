from abc import ABC, abstractmethod

from benchmark import Benchmark
from pde_solver.discretization import DiscreteSolution
from pde_solver.ode_solver import ExplicitRungeKuttaMethod
from pde_solver.system_vector import SystemVector
from pde_solver.time_stepping import TimeStepping, TimeStepTooSmallError
from tqdm.auto import tqdm


class Solver(ABC):
    name: str
    short: str
    problem: str
    benchmark: Benchmark
    solution: DiscreteSolution
    time_stepping: TimeStepping
    ode_solver: ExplicitRungeKuttaMethod
    _right_hand_side: SystemVector

    @abstractmethod
    def __init__(self, problem: str, benchmark: Benchmark, **kwargs):
        ...

    @property
    def right_hand_side(self) -> SystemVector:
        return self._right_hand_side

    @right_hand_side.setter
    def right_hand_side(self, right_hand_side: SystemVector):
        self._right_hand_side = right_hand_side
        self._setup_ode_solver()

    def _setup_ode_solver(self):
        self.ode_solver.time = self.solution.start_time
        self.ode_solver.start_value = self.solution.initial_data
        self.ode_solver.right_hand_side = self.right_hand_side

    def solve(self, leave_progress_bar=True):
        try:
            for time_step in tqdm(
                self.time_stepping,
                desc=self.name,
                postfix={"DOFs": self.solution.dimension},
                leave=leave_progress_bar,
            ):
                self.update(time_step)

        except TimeStepTooSmallError:
            tqdm.write("WARNING: time step is too small calculation is interrupted")

    def update(self, time_step: float):
        self.ode_solver.execute(time_step)
        self.solution.add_solution(time_step, self.ode_solver.solution)

    def __repr__(self) -> str:
        return f"{self.name}(short={self.short}, right_side={self.right_hand_side}, ode_solver={self.ode_solver}, time_stepping={self.time_stepping})"
