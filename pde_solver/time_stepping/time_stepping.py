from abc import ABC, abstractmethod

from pde_solver.mesh import Mesh


class TimeStepTooSmallError(Exception):
    ...


class TimeStepping(ABC):
    start_time: float
    end_time: float
    cfl_number: float

    time: float
    time_steps: int

    """Iterator for time stepping used for solving PDEs.

    Usage: delta_t in time_stepping_object

    where TIME_STEPPING_OBJECT is an object of a subclass of 'TimeStepping'.

    """
    _stop_iteration = False

    def __iter__(self):
        self._stop_iteration = False
        self.update_time_step()
        self.time = self.start_time
        self.time_steps = 0
        return self

    def update_time_step(self):
        # can be modified by subclass
        ...

    def __next__(self):
        if self._stop_iteration:
            raise StopIteration

    @property
    def time_step(self) -> float:
        time_step = min(self.desired_time_step, self.end_time - self.time)
        self._update_time(time_step)

        return time_step

    @property
    @abstractmethod
    def desired_time_step(self) -> float:
        ...

    def _update_time(self, time_step: float):
        if time_step < 1e-12:
            raise TimeStepTooSmallError(f"time step {time_step} is too small.")

        self.time += time_step
        self.time_steps += 1

        if self.end_time - self.time < 1e-12:
            self.time = self.end_time
            self._stop_iteration = True

    def satisfy_cfl(self) -> bool:
        # can be modified by subclasses
        return True


class SpatialMeshDependendetTimeStepping(TimeStepping):
    """This time stepping ensures that the time step width dt is approximately
    CFL_NUMBER*h, where h is the mesh width of a given spatial mesh. At the end
    dt could be smaller to reach end time.

    """

    _mesh_width: float

    @property
    def mesh(self):
        ...

    @mesh.setter
    def mesh(self, mesh: Mesh):
        self._mesh_width = mesh.step_length

    @property
    def desired_time_step(self) -> float:
        return self.cfl_number * self._mesh_width
