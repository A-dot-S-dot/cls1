from pde_solver.time_stepping import *


class TimeSteppingFactory:
    problem_name: str
    adaptive = False

    @property
    def mcl_time_stepping(self) -> MCLTimeStepping:
        if self.problem_name == "advection":
            return AdvectionMCLTimeStepping()
        else:
            if self.adaptive:
                return AdaptiveMCLTimeStepping()
            else:
                return StaticMCLTimeStepping()

    @property
    def mesh_time_stepping(self) -> SpatialMeshDependendetTimeStepping:
        return SpatialMeshDependendetTimeStepping()
