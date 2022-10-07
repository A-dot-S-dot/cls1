import pde_solver
from pde_solver.ode_solver import ExplicitRungeKuttaMethod
from pde_solver.system_vector import SystemVector


class ScalarFiniteElementSolver(pde_solver.PDESolver):
    right_hand_side: SystemVector
    ode_solver: ExplicitRungeKuttaMethod

    def update(self):
        time_step = self.time_stepping.time_step

        self.ode_solver.execute(time_step)
        self.solution.add_solution(time_step, self.ode_solver.solution)
