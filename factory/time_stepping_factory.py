from pde_solver.time_stepping import *


class TimeSteppingFactory:
    problem_name: str

    @property
    def mcl_time_stepping(self) -> MCLTimeStepping:
        if self.problem_name == "advection":
            return ConstantMCLTimeStepping()
        else:
            return AdaptiveMCLTimeStepping()

    @property
    def mesh_time_stepping(self) -> SpatialMeshDependendetTimeStepping:
        return SpatialMeshDependendetTimeStepping()


TIME_STEPPING_FACTORY = TimeSteppingFactory()
