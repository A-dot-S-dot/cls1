from abc import ABC, abstractmethod

from mesh import Mesh
from system.matrix.system_matrix import SystemMatrix
from system.vector.system_vector import SystemVector
from tqdm import tqdm


class TimeStepping(ABC):
    start_time: float
    end_time: float

    _cfl_number: float
    _time: float

    """Iterator for time stepping used for solving PDEs.

    Usage: delta_t in time_stepping_object

    where TIME_STEPPING_OBJECT is an object of a subclass of 'TimeStepping'.

    """

    def __iter__(self):
        self._time = self.start_time
        return self

    @property
    def cfl_number(self) -> float:
        return self._cfl_number

    @cfl_number.setter
    def cfl_number(self, cfl_number: float):
        assert cfl_number > 0, f"cfl number {cfl_number} is not positive"
        self._cfl_number = cfl_number

    @property
    @abstractmethod
    def delta_t(self) -> float:
        ...

    def __next__(self):
        if self._time < self.end_time:
            delta_t = min(self.delta_t, self.end_time - self._time)

            if delta_t < 1e-12:
                self._time = self.end_time
                raise StopIteration
            else:
                self._time += delta_t

            return delta_t
        else:
            raise StopIteration


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
    def delta_t(self) -> float:
        return self._cfl_number * self._mesh_width


class MCLTimeStepping(TimeStepping):
    """MCL time stepping base class."""

    lumped_mass: SystemVector
    artificial_diffusion: SystemMatrix

    _delta_t: float

    def set_delta_t(self):
        self._delta_t = self._cfl_number * min(
            self.lumped_mass.values / abs(self.artificial_diffusion.values.diagonal())
        )


class AdvectionMCLTimeStepping(MCLTimeStepping):
    """This time stepping ensures that the time step for linear advection solvers is

        dt = CFL_NUMBER*min(mi/dii)

    where mi denotes the lumped mass and dij a diffusion. At the end dt could be
    smaller to reach end time.

    """

    @property
    def delta_t(self) -> float:
        return self._delta_t


class StaticMCLTimeStepping(MCLTimeStepping):
    """This time stepping checks that the time step dt (which is initialized at the beginning), suffices

        dt < CFL_NUMBER*min(mi/dii)

    where mi denotes the lumped mass and dij a diffusion.
    """

    @property
    def delta_t(self) -> float:
        if self._delta_t < min(
            self.lumped_mass.values / abs(self.artificial_diffusion.values.diagonal())
        ):
            tqdm.write("WARNING: CFL condition violated")
        return self._delta_t


class AdaptiveMCLTimeStepping(MCLTimeStepping):
    """This time stepping ensures that the time step

        dt = CFL_NUMBER*min(mi/dii)

    where mi denotes the lumped mass and dij a diffusion. At the end dt could be
    smaller to reach end time.

    """

    @property
    def delta_t(self) -> float:
        self.set_delta_t()
        return self._delta_t
