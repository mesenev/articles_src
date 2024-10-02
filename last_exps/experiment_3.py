# from dolfin import *
#
from last_exps.draw_graphics import surf_draw
# from utilities import print_2d_isolines, print_2d
# from dolfin import dx, ds
#
# parameters["form_compiler"]["optimize"] = True
# parameters["form_compiler"]["cpp_optimize"] = True
# set_log_active(True)
#
# omega3d = UnitSquareMesh(20, 20)
# finite_element = FiniteElement("CG", omega3d.ufl_cell(), 1)
# square = FunctionSpace(omega3d, finite_element)
# state_space = FunctionSpace(omega3d, finite_element * finite_element)
# simple_space = FunctionSpace(omega3d, finite_element)
# vector_space = VectorFunctionSpace(omega3d, 'CG', 1)
#
# _lambda = 0.1 ** 2
# v, h = TestFunctions(state_space)
# epsilon = 0.1 ** 10
#
# a = 0.6
# alpha = 0.333
# ka = 1
# b = 0.025
# beta = 1
# state = Function(state_space)
# theta, phi = split(state)
#
# theta_b = project(Expression('x[1]', degree=2), simple_space)
# theta_b_4 = project(Expression('pow(t, 4)', degree=2, t=theta_b, ), simple_space)
# gamma = Constant(0.1)
#
#
# def solve_boundary():
#     theta_equation = \
#         a * inner(grad(theta), grad(v)) * dx \
#         + a * theta * v * ds + \
#         + b * ka * inner(theta ** 4 - phi, v) * dx
#     theta_src = beta * theta_b_4 * v * ds
#     phi_equation = \
#         alpha * inner(grad(phi), grad(h)) * dx \
#         + alpha * phi * h * ds \
#         + ka * inner(phi - theta ** 4, h) * dx
#     phi_src = gamma * theta_b_4 * h * ds
#     solve(
#         theta_equation + phi_equation - theta_src - phi_src == 0, state,
#         form_compiler_parameters={"optimize": True, 'quadrature_degree': 3},
#         solver_parameters={"newton_solver": {"linear_solver": "mumps"}}
#     )
#     return state.split()
#
#
# point = Point(0.5, 0.5)
# phi_ans_gamma_dynamic = list()
# theta_ans_gamma_dynamic = list()
# for i in range(50):
#     gamma = Constant(0.1 + 0.8 * (i / 100))
#     for j in range(50):
#         print(i, j)
#         beta = Constant(1 + 1 * (j / 100))
#         theta_ans, phi_ans = solve_boundary()
#
#         theta_ans_gamma_dynamic.append(
#             (0.1 + 0.8 * (i / 100), 1 + 1 * (j / 100), theta_ans(point))
#         )
#         phi_ans_gamma_dynamic.append(
#             (0.1 + 0.8 * (i / 100), 1 + 1 * (j / 100), phi_ans(point))
#         )
#
# with open("scratch/theta_ans_gamma_dynamic.txt", "w") as file:
#     print(*theta_ans_gamma_dynamic, file=file)
#
# with open("scratch/phi_ans_gamma_dynamic.txt", "w") as file:
#     print(*phi_ans_gamma_dynamic, file=file)
#
surf_draw('theta')
surf_draw('phi')
