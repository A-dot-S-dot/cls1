from core import FiniteVolumeSpace, Interval, UniformMesh

domain = Interval(0, 1)
VOLUME_MESH = UniformMesh(domain, 4)
VOLUME_SPACE = FiniteVolumeSpace(VOLUME_MESH)
