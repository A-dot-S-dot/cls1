from pde_solver.discretization.finite_volume import FiniteVolumeSpace
from pde_solver.mesh import Interval, UniformMesh

domain = Interval(0, 1)
VOLUME_MESH = UniformMesh(domain, 4)
VOLUME_SPACE = FiniteVolumeSpace(VOLUME_MESH)
