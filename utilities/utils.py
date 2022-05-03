# file should contain math utilities for operating functions
import itertools
import os
import shutil
from abc import ABC
from os import mkdir

from dolfin import *
from dolfin import ds
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from numpy import cross, sqrt, newaxis, vectorize


def clear_dir(folder):
    try:
        shutil.rmtree(f'{folder}_prev')
    except OSError:
        print(f"Deletion of the directory {folder}_prev is failed.")
    try:
        os.rename(folder, folder + '_prev')
    except OSError:
        print(f"Renaming the directory {folder} to {folder}_prev is failed.")
    mkdir(folder)


def get_trace(v, steps=100):
    left = list(map(lambda x: v(Point(0, x)), (1 / (steps - 1) * _ for _ in range(0, steps))))
    top = list(map(lambda x: v(Point(x, 1)), (1 / (steps - 1) * _ for _ in range(0, steps))))
    right = list(map(lambda x: v(Point(1, x)), (1 - 1 / (steps - 1) * _ for _ in range(0, steps))))
    bottom = list(map(lambda x: v(Point(x, 0)), (1 - 1 / (steps - 1) * _ for _ in range(0, steps))))

    return left[:-1] + top[:-1] + right[:-1] + bottom


def function2d_dumper(v, folder, name):
    import numpy as np
    import codecs, json
    x, y = np.meshgrid(np.arange(0, 1.0, 0.02), np.arange(0, 1.0, 0.02))
    answer = vectorize(lambda _, __: v(Point(_, __)))(x, y)
    json.dump(
        answer.tolist(), codecs.open(f"{folder}/{name}", 'w', encoding='utf-8'),
        separators=(',', ':'), sort_keys=True, indent=1
    )
    return answer


class Normal(UserExpression, ABC):
    def __init__(self, mesh, dimension, **kwargs):
        self.mesh = mesh
        self.dimension = dimension
        super().__init__(**kwargs)

    def eval_cell(self, values, x, ufc_cell):
        values[0] = 0
        values[1] = 0
        for i in range(self.dimension):

            if x[i] < DOLFIN_EPS:
                values[i] = -1
            if x[i] + DOLFIN_EPS > 1:
                values[i] = 1
        for i, j in itertools.combinations(list(range(self.dimension)), 2):
            if abs(values[i]) == abs(values[j]) == 1:
                values[0] *= 0.5
                values[1] *= 0.5

    def value_shape(self):
        return (2,) if self.dimension == 2 else (3,)


class NormalDerivativeZ(UserExpression):
    def __init__(self, func, vector_space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = project(grad(func), vector_space)

    def eval(self, value, x):
        value[0] = self.func(x[0], x[1], 1)[2]

    def __floordiv__(self, other):
        pass


def get_facet_normal(bmesh):
    # https://bitbucket.org/fenics-project/dolfin/issues/53/dirichlet-boundary-conditions-of-the-form
    """Manually calculate FacetNormal function"""

    if not bmesh.type().dim() == 2:
        raise ValueError('Only works for 2-D mesh')

    vertices = bmesh.coordinates()
    cells = bmesh.cells()

    vec1 = vertices[cells[:, 1]] - vertices[cells[:, 0]]
    vec2 = vertices[cells[:, 2]] - vertices[cells[:, 0]]

    normals = cross(vec1, vec2)
    normals /= sqrt((normals ** 2).sum(axis=1))[:, newaxis]

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


def normal_for_square_mesh():
    mesh = UnitSquareMesh(100, 100)
    n = FacetNormal(mesh)
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * ds
    l = inner(n, v) * ds
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    nh = Function(V)

    solve(A, nh.vector(), L)
    File("nh.xml") << nh
    return nh


normal_function = []


def get_normal_derivative(function):
    if not normal_function:
        normal_function.append(normal_for_square_mesh())
    normal = normal_function[0]
    mesh = UnitSquareMesh(100, 100)
    f = FunctionSpace(mesh, "CG", 1)
    function = interpolate(function, f)
    V = VectorFunctionSpace(mesh, "CG", 1)
    return project(inner(normal, project(grad(project(function, f)), V)), f)


# normal_derivative_for_square_mesh()


def normal_for_cube_mesh():
    mesh = UnitCubeMesh(40, 40, 40)
    n = FacetNormal(mesh)
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * ds
    l = inner(n, v) * ds
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)

    A.ident_zeros()
    nh = Function(V)

    solve(A, nh.vector(), L)
    return nh


normal_function_3d = []


def get_normal_derivative_3d(function, f_space, v_space):
    if not normal_function_3d:
        normal_function.append(normal_for_cube_mesh())
    normal = normal_function[0]
    function = interpolate(function, f_space)
    return project(inner(normal, project(grad(function), v_space)), f_space)
