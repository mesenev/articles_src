import numpy as np
from dolfin import *
import matplotlib.pyplot as plt


def get_facet_normal(bmesh):
    """Manually calculate FacetNormal function"""

    # if not bmesh.type().dim() == 2:
    #     raise ValueError('Only works for 2-D mesh')

    vertices = bmesh.coordinates()
    cells = bmesh.cells()

    vec1 = vertices[cells[:, 1]] - vertices[cells[:, 0]]
    vec2 = vertices[cells[:, 2]] - vertices[cells[:, 0]]

    normals = np.cross(vec1, vec2)
    normals /= np.sqrt((normals ** 2).sum(axis=1))[:, np.newaxis]

    # Ensure outward pointing normal
    bmesh.init_cell_orientations(Expression(('x[0]', 'x[1]', 'x[2]'), degree=3))
    normals[bmesh.cell_orientations() == 1] *= -1

    V = VectorFunctionSpace(bmesh, 'DG', 0)
    norm = Function(V)
    nv = norm.vector()

    for n in (0, 1, 2):
        dofmap = V.sub(n).dofmap()
        for i in range(dofmap.global_dimension()):
            dof_indices = dofmap.cell_dofs(i)
            assert len(dof_indices) == 1
            nv[dof_indices[0]] = normals[i, n]

    return norm


mesh = UnitCubeMesh(10, 10, 10)
bmesh = BoundaryMesh(mesh, 'exterior')
n = get_facet_normal(bmesh)

# Check values on facet midpoints
for f in facets(bmesh):
    p = f.midpoint()
    f.index(), [v.index() for v in vertices(f)], n(p)

# Check correctness of mean value
area = assemble(Constant(1.) * dx(bmesh))
nds = assemble(inner(n, n) * dx)
plt.figure()
plot(n)
plt.savefig('normal.png')
