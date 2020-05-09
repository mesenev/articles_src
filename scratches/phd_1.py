'''
int2d(mainMesh)(a * Grad(theta)
' * Grad(v))
+ int2d(mainMesh)(4 * b * ka * v * thetaold ^ 3 * theta)
- int2d(mainMesh)(3 * b * ka * v * thetaold ^ 4)
- int2d(mainMesh)(b * ka * v * phi)
+ int1d(mainMesh)(beta * v * theta)
- int1d(mainMesh)(beta * v * thetaB)
//
+ int2d(mainMesh)(alpha * Grad(phi)
' * Grad(w))
+ int1d(mainMesh, gamma01)(gamma * phi * w)
+ int1d(mainMesh, gamma02)(gamma * phi * w)
+ int1d(mainMesh, gamma2)(gamma * phi * w)
+ int2d(mainMesh)(ka * phi * w)
- int2d(mainMesh)(4 * ka * thetaold ^ 3 * theta * w)
+ int2d(mainMesh)(3 * ka * thetaold ^ 4 * w) \
+ int1d(mainMesh, gamma1)(u * phi * w) \
- int1d(mainMesh, gamma1)(thetaB ^ 4 * u * w) \
- int1d(mainMesh, gamma01)(gamma * thetaB ^ 4 * w) \
- int1d(mainMesh, gamma02)(gamma * thetaB ^ 4 * w) \
- int1d(mainMesh, gamma2)(gamma * thetaB ^ 4 * w);
'''

# noinspection PyUnresolvedReferences
from dolfin import (
    FunctionSpace,
    Function, split, TestFunctions, FiniteElement, dx, ds,
    Expression, inner, grad, solve, Constant,
    NonlinearVariationalProblem, NonlinearVariationalSolver,
    derivative, VectorFunctionSpace, BoundaryMesh,
    project, interpolate, dot, assemble
)
from dolfin.cpp.generation import UnitSquareMesh

from scratches.utilities import print_2d_boundaries, Normal

a = 0.006
alpha = 0.333
ka = 1
b = 0.025
beta = 1
gamma = 0.5
theta_0 = Expression("x[0] * sin(x[1]) + 0.1", degree=2)

omega = UnitSquareMesh(32, 32)
finite_element = FiniteElement("Lagrange", omega.ufl_cell(), 2)
omega_b = BoundaryMesh(omega, 'exterior')
boundary_space = FunctionSpace(omega_b, 'Lagrange', 3)
boundary_vector_space = VectorFunctionSpace(omega_b, 'Lagrange', 1)
state_space = FunctionSpace(omega, finite_element * finite_element)
vector_space = VectorFunctionSpace(omega, 'Lagrange', 2)
simple_space = FunctionSpace(omega, finite_element)

n = project(Normal(omega), boundary_vector_space)

v, h = TestFunctions(state_space)
state = Function(state_space)
theta, phi = split(state)

boundary_problem = \
    a * inner(grad(theta), grad(v)) * dx \
    + alpha * inner(grad(phi), grad(h)) * dx \
    + b * ka * inner(theta ** 4 - phi, v) * dx \
    + ka * inner(phi - theta ** 4, h) * dx \
    + beta * inner(theta - theta_0, v) * ds \
    + gamma * inner(phi - theta_0 ** 4, h) * ds

jcb = derivative(boundary_problem, state)
problem = NonlinearVariationalProblem(boundary_problem, state, J=jcb)
solver = NonlinearVariationalSolver(problem)
solver.solve()
answer = state.split()

grad_theta = project(grad(answer[0]), vector_space)
dn_theta = project(dot(n, interpolate(grad_theta, boundary_vector_space)), boundary_space)
zero = project(a * dn_theta + interpolate(answer[0], boundary_space) - theta_0, boundary_space)
print_2d_boundaries(zero, terminal_only=True)
