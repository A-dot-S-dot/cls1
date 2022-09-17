from fem.lagrange.lagrange import LagrangeFiniteElementSpace
from mesh import Interval
from mesh.uniform import UniformMesh
from system.vector.dof_vector import DOFVector

domain = Interval(0, 1)
LINEAR_MESH = UniformMesh(domain, 4)
LINEAR_LAGRANGE_SPACE = LagrangeFiniteElementSpace(LINEAR_MESH, 1)
LINEAR_DOF_VECTOR = DOFVector(LINEAR_LAGRANGE_SPACE)

QUADRATIC_MESH = UniformMesh(domain, 2)
QUADRATIC_LAGRANGE_SPACE = LagrangeFiniteElementSpace(QUADRATIC_MESH, 2)
QUADRATIC_DOF_VECTOR = DOFVector(QUADRATIC_LAGRANGE_SPACE)
