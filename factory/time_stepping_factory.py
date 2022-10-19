from pde_solver.time_stepping import *
from pde_solver.discrete_solution import DiscreteSolutionObservable


class TimeSteppingFactory:
    problem_name: str

    @property
    def mesh_time_stepping(self) -> SpatialMeshDependendetTimeStepping:
        return SpatialMeshDependendetTimeStepping()

    def get_godunov_time_stepping(self, adaptive=False) -> SWEGodunovTimeStepping:
        if adaptive:
            return SWEGodunovTimeStepping()
        else:
            return SWEGodunovConstantTimeStepping()

    def get_mcl_time_stepping(
        self, discrete_solution: DiscreteSolutionObservable
    ) -> MCLTimeStepping:
        if self.problem_name == "advection":
            return MCLTimeStepping()
        else:
            return AdaptiveMCLTimeStepping(discrete_solution)


TIME_STEPPING_FACTORY = TimeSteppingFactory()
