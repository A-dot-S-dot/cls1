from core import LagrangeSpace, Interval, UniformMesh

domain = Interval(0, 1)
LINEAR_MESH = UniformMesh(domain, 4)
LINEAR_LAGRANGE_SPACE = LagrangeSpace(LINEAR_MESH, 1)

QUADRATIC_MESH = UniformMesh(domain, 2)
QUADRATIC_LAGRANGE_SPACE = LagrangeSpace(QUADRATIC_MESH, 2)
