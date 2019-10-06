import matplotlib.pyplot as plt
from dolfin import *

from phd_2.experiments.normal import Normal
from phd_2.experiments.utilities import print_2d_boundaries

mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)


def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS


u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Expression("x[0] + x[1]", degree=2)
g = Expression("1", degree=2)
a = inner(grad(u), grad(v)) * dx
L = f * v * dx + g * v * ds
u = Function(V)
solve(a == L, u, bc)
n = Normal(mesh=mesh, degree=2)
u_n = project(dot(n, grad(u)), V)
for i in range(1, 10):
    p = Point(1./i, 0)
    print(u_n(p) - g(p))
# u_n = project(g, V)
# print_2d_boundaries(u, name='u_b')
# u_n = project(dot(grad(u), n), V)
# print_2d_boundaries(f, name='f')
