from base.discretization.finite_volume import FiniteVolumeSpace
from base.mesh import Interval, UniformMesh

domain = Interval(0, 1)
VOLUME_MESH = UniformMesh(domain, 4)
VOLUME_SPACE = FiniteVolumeSpace(VOLUME_MESH)
