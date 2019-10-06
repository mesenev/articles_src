import matplotlib.pyplot as plt
from dolfin import *
from dolfin import dx, ds

from phd_2.experiments.normal import Normal
from phd_2.experiments.utilities import print_2d_boundaries

a = 0.6
alpha = 0.33
k_a = 1
b = 0.025
theta_b = Expression("x[0]*cos(x[1])/2", degree=2)
q_b = 0.1


def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS or \
           x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS


omega = UnitSquareMesh(50, 50)
omega_b = BoundaryMesh(omega, 'exterior')

boundary_space = FunctionSpace(omega_b, 'Lagrange', 3)

state_space = FunctionSpace(omega, 'Lagrange', 3)
bc = DirichletBC(state_space, theta_b, boundary)
n = FacetNormal(omega)

theta = Function(state_space)
phi = Function(state_space)
v = TestFunction(state_space)

F = a * inner(grad(theta), grad(v)) * dx \
    + b * k_a * inner(theta ** 4, v) * dx \
    + a * k_a / alpha * inner(theta, v) * dx
    # + a * q_b * v * ds

solve(F == 0, theta, bc,
      solver_parameters={"newton_solver": {
          "relative_tolerance": 1e-6,
          "maximum_iterations": 50,
      }})

phi = project(- a * div(grad(theta)) / (b * k_a) + theta ** 4, state_space)
n = Normal(mesh=mesh, degree=2)
theta_n = project(dot(n, grad(theta)), state_space)
plt.figure()
print_2d_boundaries(theta_n, name='theta_n')
plot(theta_n, title="Solution", scalarbar=True)
# plt.savefig('full.png')
