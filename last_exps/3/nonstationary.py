from dolfin import *
# noinspection PyUnresolvedReferences
from dolfin import dx, ds

from dolfin import dx, ds, assemble
from utilities import clear_dir

from matplotlib import tri
from dolfin import *
from matplotlib import pyplot as plt

from phd_3.experiments.default_values import DefaultValues2D
from phd_3.experiments.solver import Problem

from phd_3.experiments.consts import DIRICHLET, NEWMAN

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
solver_params = {"newton_solver": {
    "maximum_iterations": 50,
    # "relative_tolerance": 1e-4,
    # "absolute_tolerance": 1e-5,
}}

a = 0.6
alpha = 0.333
ka = 1
b = 0.025
beta = 1
gamma = 1
r = None
lmbd = 1
epsilon = 0.1 ** 10

omega = UnitSquareMesh(75, 75)
omega_b = BoundaryMesh(omega, 'exterior')
finite_element = FiniteElement("CG", omega.ufl_cell(), 1)

state_space = FunctionSpace(omega, finite_element * finite_element)
simple_space = FunctionSpace(omega, finite_element)
vector_space = VectorFunctionSpace(omega, 'CG', 1)

v, h = TestFunctions(state_space)
state = Function(state_space)
theta, phi = split(state)
sigma = Constant(0.99)

theta_b = project(Expression('0.2 + x[1] / 2', degree=2), simple_space)
theta_b_4 = project(Expression('pow(t, 4)', degree=2, t=theta_b,), simple_space)
theta_in = project(Expression('0.8 * cos(x[1]*10)', degree=2), simple_space)

theta_equation = a * inner(grad(theta), grad(v)) * dx \
                 + a * theta * v * ds + \
                 + b * ka * inner(theta ** 4 - phi, v) * dx
theta_src = inner(a * (theta_b + theta_n), v) * ds

phi_equation = \
    alpha * inner(grad(phi), grad(h)) * dx \
    + alpha * phi * h * ds \
    + ka * inner(phi - theta ** 4, h) * dx
phi_src = phi_n * h * ds
solve(
    theta_equation + phi_equation - theta_src - phi_src == 0, state,
    form_compiler_parameters={"optimize": True, 'quadrature_degree': 3}
)
