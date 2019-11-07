import numpy as np
from dolfin import *

mesh = UnitSquareMesh(10, 10)
bmesh = BoundaryMesh(mesh, 'exterior')


class Normal(UserExpression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)

    def eval_cell(self, values, x, ufc_cell):
        values[0] = 0
        values[1] = 0
        if x[0] < DOLFIN_EPS:
            values[0] = -1
        if x[0] + DOLFIN_EPS > 1:
            values[0] = 1
        if x[1] < DOLFIN_EPS:
            values[1] = -1
        if x[1] + DOLFIN_EPS > 1:
            values[1] = 1
        if abs(values[0]) == abs(values[1]) == 1:
            values[0] *= 0.65
            values[1] *= 0.65

    def value_shape(self):
        return (2,)


def get_facet_normal(bmesh):
    # https://bitbucket.org/fenics-project/dolfin/issues/53/dirichlet-boundary-conditions-of-the-form
    '''Manually calculate FacetNormal function'''

    if not bmesh.type().dim() == 2:
        raise ValueError('Only works for 2-D mesh')

    vertices = bmesh.coordinates()
    cells = bmesh.cells()

    vec1 = vertices[cells[:, 1]] - vertices[cells[:, 0]]
    vec2 = vertices[cells[:, 2]] - vertices[cells[:, 0]]

    normals = np.cross(vec1, vec2)
    normals /= np.sqrt((normals ** 2).sum(axis=1))[:, np.newaxis]

    # Ensure outward pointing normal
    bmesh.init_cell_orientations(Expression(('x[0]', 'x[1]', 'x[2]')))
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
