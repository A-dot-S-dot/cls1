from collections.abc import Iterator
from mesh import Mesh


class TimeStepping(Iterator):
    _start_time: float

    """Iterator for time stepping used for solving PDEs.

    Usage: delta_t in time_stepping_object

    where TIME_STEPPING_OBJECT is an object of a subclass of 'TimeStepping'.

    """

    @property
    def start_time(self) -> float:
        return self._start_time

    def __iter__(self):
        self._time = self._start_time
        return self


class SpatialMeshDependendetTimeStepping(TimeStepping):
    """This time stepping ensures that the time step width dt is approximately
    CFL_NUMBER*h, where h is the mesh width of a given spatial mesh.

    """

    _start_time: float
    _end_time: float
    _mesh_width: float
    _cfl_number: float
    _reached_end_time: bool

    def __init__(
        self, start_time: float, end_time: float, mesh: Mesh, cfl_number: float
    ):
        assert cfl_number > 0, f"cfl number {cfl_number} is not positive"
        self._start_time = start_time
        self._end_time = end_time
        self._mesh_width = mesh.step_length
        self._cfl_number = cfl_number

    def __next__(self):
        if self._time < self._end_time:
            delta_t = min(
                self._cfl_number * self._mesh_width, self._end_time - self._time
            )

            if delta_t < 1e-12:
                self._time = self._end_time
            else:
                self._time += delta_t

            return delta_t
        else:
            raise StopIteration
