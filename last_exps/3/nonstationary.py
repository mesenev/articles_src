'''Оценка влияния скачка коэфициента теплопроводности на тепловое излучение'''
from time import sleep

from dolfin import *
# noinspection PyUnresolvedReferences
from dolfin import dx, ds
from dolfin import dx, ds
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import pyplot

from simple import draw_simple

from utilities import print_two_with_colorbar

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
solver_params = {"newton_solver": {
    "maximum_iterations": 50,
}}
set_log_active(True)

a = 0.6
alpha = 0.333
ka = 1
b = 0.025
beta = 1
gamma = 1
p = 1
lmbd = 1
epsilon = 0.1 ** 10
sigma = 60
m_param = 1

omega = UnitSquareMesh(30, 30)
omega_b = BoundaryMesh(omega, 'exterior')
finite_element = FiniteElement("CG", omega.ufl_cell(), 1)

state_space = FunctionSpace(omega, finite_element * finite_element)
simple_space = FunctionSpace(omega, finite_element)
vector_space = VectorFunctionSpace(omega, 'CG', 1)

v, h = TestFunctions(state_space)
state = Function(state_space)
theta, phi = split(state)

theta_b = project(Expression('1', degree=2), simple_space)
theta_b_4 = project(Expression('pow(t, 4)', degree=2, t=theta_b, ), simple_space)
theta_0 = project(Expression('0', degree=2), simple_space)

theta_f_b = p * inner(theta_b, v) * ds
phi_g_b = gamma * inner(theta_b_4, h) * ds
theta_f = project(Expression("0", degree=2), simple_space)
phi_g = project(Expression("0", degree=2), simple_space)
theta_src = theta_f_b + theta_f * v * dx
phi_src = phi_g_b + phi_g * h * dx
theta_star = 0.2
theta_prev = theta_0


def sigma_src():
    class K_theta_param(UserExpression):
        def eval(self, values, x):
            values[0] = m_param if theta_prev(x) > theta_star else 0.1

    return K_theta_param()


k_theta = project(sigma_src(), simple_space)

theta_equation = (
        sigma * inner(theta - theta_prev, v) * dx
        + k_theta * inner(grad(theta), grad(v)) * dx
        + p * theta * v * ds
        + b * ka * inner(theta ** 4 - phi, v) * dx
)

phi_equation = \
    alpha * inner(grad(phi), grad(h)) * dx \
    + gamma * phi * h * ds \
    + ka * inner(phi - theta ** 4, h) * dx
answers = list()


def main(n: int):
    images = list()
    theta_avg = list()
    theta_avg.append(assemble(theta_0 * dx))
    for j in range(1):
        for i in range(1200):
            solve(
                theta_equation + phi_equation - theta_src - phi_src == 0, state,
                form_compiler_parameters={"optimize": True, 'quadrature_degree': 3}
            )
            theta, phi = state.split()
            a = phi.vector()
            b = theta.vector()
            theta_prev.interpolate(theta)
            k_theta.interpolate(sigma_src())
            print(m_param, a.min(), a.max(), b.min(), b.max())
            theta_avg.append(assemble(theta * dx))

            answers.append((m_param, a.min(), a.max(), b.min(), b.max()))
            print_two_with_colorbar(theta, phi, f"{i}", folder="animation")

    with open(f"theta_star/data{n}.txt", "w") as f:
        print(*answers, sep='\n', file=f)

    with open(f"theta_star/theta_avg{n}.txt", "w") as f:
        print(*theta_avg, sep='\n', file=f)





if __name__ == "__main__":
    main(1)
    theta_prev.interpolate(theta_0)
    k_theta.interpolate(sigma_src())
    theta_star = 0.6
    main(2)
