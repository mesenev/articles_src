from dolfin import *

from utilities import print_2d_isolines, print_2d
from dolfin import dx, ds

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
set_log_active(True)

omega2d = UnitSquareMesh(50, 50)
finite_element = FiniteElement("CG", omega2d.ufl_cell(), 1)
square = FunctionSpace(omega2d, finite_element)
state_space = FunctionSpace(omega2d, finite_element * finite_element)
simple_space = FunctionSpace(omega2d, finite_element)
# boundary_vector_space = VectorFunctionSpace(omega_b, 'CG', 1)
vector_space = VectorFunctionSpace(omega2d, 'CG', 1)
# boundary_simple_space = FunctionSpace(omega_b, 'CG', 1)

_lambda = 0.1 ** 2
v, h = TestFunctions(state_space)
epsilon = 0.1 ** 10

a = 0.6
alpha = 0.333
ka = 1
b = 0.025
beta = 1
state = Function(state_space)
theta, phi = split(state)

theta_b = project(Expression('0.2 + x[1] / 2', degree=2), simple_space)
theta_b_4 = project(Expression('pow(t, 4)', degree=2, t=theta_b,), simple_space)
gamma = Expression("0.8 * cos(x[1]*3.14/2) + 0.5", degree=2)


def solve_boundary():
    theta_equation = \
        a * inner(grad(theta), grad(v)) * dx \
        + a * theta * v * ds + \
        + b * ka * inner(theta ** 4 - phi, v) * dx
    theta_src = beta * theta_b_4 * v * ds
    phi_equation = \
        alpha * inner(grad(phi), grad(h)) * dx \
        + alpha * phi * h * ds \
        + ka * inner(phi - theta ** 4, h) * dx
    phi_src = gamma * theta_b_4 * h * ds
    solve(
        theta_equation + phi_equation - theta_src - phi_src == 0, state,
        form_compiler_parameters={"optimize": True, 'quadrature_degree': 3},
        solver_parameters={"newton_solver": {"linear_solver": "mumps"}}
    )
    return state.split()


theta_ans, phi_ans = solve_boundary()
# to_print = function2d_dumper(
#     lambda p: abs(theta_n_diff(Point(p[0], p[1], 1))),
#     folder='scratch', name=target
# )
print_2d_isolines(
    theta_ans, name='theta_iso', folder='scratch',
    # table=True,
    # levels=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.05, 0.1]
)
print_2d(theta_ans, name='theta', folder='scratch',)
print_2d(phi_ans, name='phi', folder='scratch',)
print_2d_isolines(phi_ans, name='phi_iso', folder='scratch', )
