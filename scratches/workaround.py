from dolfin import Expression, VectorFunctionSpace, UnitCubeMesh, BoundaryMesh, \
    facets, vertices, dx, ds, assemble, Constant, inner, plot, dot, Function
import numpy as np
from matplotlib import pyplot as plt


def get_facet_normal(bmesh):
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


mesh = UnitCubeMesh(5, 5, 5)
bmesh = BoundaryMesh(mesh, 'exterior')
n = get_facet_normal(bmesh)

# Check values on facet midpoints
for f in facets(bmesh):
    p = f.midpoint()
    print(f.index(), [v.index() for v in vertices(f)], n(p))

# Check correctness of mean value
area = assemble(Constant(1.) * dx(bmesh))
nds = assemble(inner(n, n) * dx(bmesh))
print("Average value of RT1 normal on boundary:", nds / area)
f = Expression(('x[0]/2', 'x[1]*3', 'x[2]/3'), degree=3)
plot(n, title="Normal")
plt.savefig('norm.png')
