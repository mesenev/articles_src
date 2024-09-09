from dolfin import *

from utilities import print_2d_isolines, print_2d
from dolfin import dx, ds

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
set_log_active(True)

omega3d = UnitSquareMesh(50, 50)
finite_element = FiniteElement("CG", omega3d.ufl_cell(), 1)
square = FunctionSpace(omega3d, finite_element)
state_space = FunctionSpace(omega3d, finite_element * finite_element)
simple_space = FunctionSpace(omega3d, finite_element)
vector_space = VectorFunctionSpace(omega3d, 'CG', 1)

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

theta_b = project(Expression('2 * x[1]', degree=2), simple_space)
theta_b_4 = project(Expression('pow(t, 4)', degree=2, t=theta_b, ), simple_space)
gamma = Constant(0.1)


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



gamma = Constant(0.1)
theta_ans, _ = solve_boundary()
print_2d_isolines(theta_ans, name='theta_1', folder='scratch', )
print_2d_isolines(_, name='phi_1', folder='scratch', )
gamma = Constant(1)
theta_ans, _ = solve_boundary()
print_2d_isolines(theta_ans, name='theta_2', folder='scratch', )
print_2d_isolines(_, name='phi_2', folder='scratch', )


# Расклад такой -- при увеличении тетта_б мы можем повысить вклад гаммы в результирующее температурное поле
# но при этом поле излучения улетает в космос. Что неприятно (или нет, не понятно)