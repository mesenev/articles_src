import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
from dolfin import *
# noinspection PyUnresolvedReferences
from dolfin import dx, ds

from phd_2.experiments.utilities import print_2d_boundaries, get_facet_normal, points_2d, Normal, print_simple_graphic

a = 0.6
alpha = 0.33
k_a = 1
b = 0.025
theta_0 = Expression("x[0]*sin(x[1]) + 0.1", degree=2)

omega = UnitSquareMesh(120, 120)
function_space = FunctionSpace(omega, 'CG', 2)
theta = Function(function_space)
v = TestFunction(function_space)

F = - a * inner(div(grad(theta)), div(grad(v))) * dx \
    + b * k_a * inner((theta_0 + theta) ** 4, div(grad(v))) * dx \
    + a * k_a / alpha * inner(theta_0 + theta, div(grad(v))) * dx \
    - a * inner(div(grad(project(theta_0, function_space))), div(grad(v))) * dx

solve(F == 0, theta,  # bc,
      solver_parameters={"newton_solver": {
          "relative_tolerance": 1e-12,
          "maximum_iterations": 50,
      }})
print_2d_boundaries(theta, terminal_only=True)
plt.figure()
c = plot(theta, title="theta", mode='color')
plt.colorbar(c)
plt.savefig('theta.png')
plt.figure()
c = plot(interpolate(theta_0, function_space), title="theta_0", mode='color')
plt.colorbar(c)
plt.savefig('theta_0.png')

plt.figure()
c = plot(project(theta + interpolate(theta_0, function_space), function_space), title="answer", mode='color')
plt.colorbar(c)
plt.savefig('answer.png')
