from core import Interval, UniformMesh
from core.finite_volume import FiniteVolumeSpace

domain = Interval(0, 1)
VOLUME_MESH = UniformMesh(domain, 4)
VOLUME_SPACE = FiniteVolumeSpace(VOLUME_MESH)
