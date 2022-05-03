from os import listdir
from os.path import isfile, join

from experiment3 import *

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

folder = 'exp3'
set_log_active(False)

# class NormalDerivativeZ(UserExpression):
#     def __init__(self, func, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.func = project(grad(func), DefaultValues3D.vector_space)
#
#     def eval(self, value, x):
#         value[0] = self.func(x[0], x[1], 1)[2]
#
#     def __floordiv__(self, other):
#         pass


xml_files = [f for f in listdir(folder) if isfile(join(folder, f)) and f.split('.')[1] == 'xml']
problem.solve_boundary()
iterator = problem.find_optimal_control(2)
next(iterator), next(iterator), next(iterator)
theta = Function(problem.theta.function_space(), f'{folder}/solution_final.xml')
# for i in xml_files:
#     print(i)
#     target = i.split('.')[0]
#     theta = Function(problem.theta.function_space(), f'{folder}/{i}')
    # theta_n_final = project(NormalDerivativeZ(theta), square)
    # theta_n = problem.def_values.theta_n
    # theta_n_diff = project(abs(theta_n_final - theta_n) / abs(theta_n), square)
    # to_print = function2d_dumper(
    #     lambda p: abs(theta_n_diff(Point(p[0], p[1], 1))),
    #     folder='scratch', name=target
    # )
    # print_2d_isolines(
    #     theta_n_diff, name=target + '_iso', folder=folder,
    # theta, name=target + '_iso', folder=folder,
    # table=True,
    # levels=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.05, 0.1]
    # )
    # print_2d(theta_n_diff, name=target + '_square', folder=folder, )

# with open('exp1/quality.txt', 'r') as f:
#     data = list(map(float, f.read().split()))
# draw_simple_graphic(data, name='quality', folder=folder)