from pde_solver.discretization.finite_element import LagrangeFiniteElementSpace
from pde_solver.mesh import Interval, UniformMesh

domain = Interval(0, 1)
LINEAR_MESH = UniformMesh(domain, 4)
LINEAR_LAGRANGE_SPACE = LagrangeFiniteElementSpace(LINEAR_MESH, 1)

QUADRATIC_MESH = UniformMesh(domain, 2)
QUADRATIC_LAGRANGE_SPACE = LagrangeFiniteElementSpace(QUADRATIC_MESH, 2)
