from dolfin import *

mesh = UnitSquareMesh(64, 64)

P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, P1 * R)
vector_space = VectorFunctionSpace(mesh, 'Lagrange', 2)

(u, c) = TrialFunction(W)
(v, d) = TestFunctions(W)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("-sin(5*x[0])", degree=2)
a = (inner(grad(u), grad(v)) + c * v + u * d) * dx
L = f * v * dx + g * v * ds

w = Function(W)
solve(a == L, w)
(u, c) = w.split()
grad_u = project(grad(u), vector_space)

file = File("neumann_poisson.pvd")
file << u
