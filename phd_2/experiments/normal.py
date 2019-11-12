import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
from dolfin import *
# noinspection PyUnresolvedReferences
from dolfin import dx, ds

from phd_2.experiments.utilities import print_2d_boundaries, get_facet_normal, points_2d, Normal

mesh = UnitSquareMesh(10, 10)
bmesh = BoundaryMesh(mesh, 'exterior')
a = 0.6
alpha = 0.33
k_a = 1
b = 0.025
theta_b = Expression("x[0]*cos(x[1])/2 + 0.1", degree=2)
q_b = Expression("x[0]*cos(x[1])/2 + 0.1", degree=2)


def boundary(x):
    return x[0] - DOLFIN_EPS < 0 or x[0] > 1.0 - DOLFIN_EPS or \
           x[1] - DOLFIN_EPS < 0 or x[1] > 1.0 - DOLFIN_EPS


omega = UnitSquareMesh(50, 50)
omega_b = BoundaryMesh(omega, 'exterior')

function_space = FunctionSpace(omega, 'Lagrange', 3)
vector_space = VectorFunctionSpace(omega, 'Lagrange', 2)

boundary_space = FunctionSpace(omega_b, 'Lagrange', 3)
boundary_vector_space = VectorFunctionSpace(omega_b, 'Lagrange', 1)

bc = DirichletBC(function_space, theta_b, boundary)

theta = Function(function_space)
phi = Function(function_space)
v = TestFunction(function_space)

F = - a * inner(grad(theta), grad(v)) * dx \
    + b * k_a * inner(theta ** 4, v) * dx \
    + a * k_a / alpha * inner(theta, v) * dx \
    + a * q_b * v * ds

solve(F == 0, theta,  # bc,
      solver_parameters={"newton_solver": {
          "relative_tolerance": 1e-6,
          "maximum_iterations": 50,
      }})

phi = project(- a * div(grad(theta)) / (b * k_a) + theta ** 4, function_space)
n = project(Normal(omega), boundary_vector_space)
grad_theta = project(grad(theta), vector_space)
dn_theta = project(dot(n, interpolate(grad_theta, boundary_vector_space)), boundary_space)
print(*map(lambda x: str(dn_theta(x))[:8].rjust(8, ' '), points_2d), sep='\t')
print(*map(lambda x: str(theta(x))[:8].rjust(8, ' '), points_2d), sep='\t')
print(*map(lambda x: str(theta_b(x))[:8].rjust(8, ' '), points_2d), sep='\t')
print_2d_boundaries(dn_theta, name='theta_n')
print_2d_boundaries(q_b, name='theta_n_orig')
