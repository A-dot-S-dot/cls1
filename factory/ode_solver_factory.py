from ode_solver.explicit_runge_kutta import (
    ExplicitRungeKuttaMethod,
    ForwardEuler,
    Heun,
    RungeKutta8,
    StrongStabilityPreservingRungeKutta3,
    StrongStabilityPreservingRungeKutta4,
)


class ODESolverFactory:
    def get_ode_solver(self, name: str) -> ExplicitRungeKuttaMethod:
        if name == "euler":
            return ForwardEuler()
        elif name == "heun":
            return Heun()
        elif name == "ssp3":
            return StrongStabilityPreservingRungeKutta3()
        elif name == "ssp4":
            return StrongStabilityPreservingRungeKutta4()
        else:
            raise NotImplementedError(f"ode solver {name} not implemented.")

    def get_optimal_ode_solver(self, degree: int) -> ExplicitRungeKuttaMethod:
        if degree == 1:
            return self.get_ode_solver("heun")
        elif degree == 2:
            return self.get_ode_solver("ssp3")
        elif degree == 3:
            return self.get_ode_solver("ssp4")
        elif degree >= 4 and degree <= 7:
            return RungeKutta8()
        else:
            raise NotImplementedError(
                f"No optimal solver for degree {degree} available."
            )


ODE_SOLVER_FACTORY = ODESolverFactory()
