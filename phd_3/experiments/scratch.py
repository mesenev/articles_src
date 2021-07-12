from dolfin import *

from utilities import print_2d_isolines, print_2d
from default_values import DefaultValues3D

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
set_log_active(False)
omega2d = UnitSquareMesh(50, 50)
finite_element = FiniteElement("CG", omega2d.ufl_cell(), 1)
square = FunctionSpace(omega2d, finite_element)


class NormalDerivativeZ(UserExpression):
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = project(grad(func), DefaultValues3D.vector_space)

    def eval(self, value, x):
        value[0] = self.func(x[0], 0, x[1])[1]


theta_n = Expression('x[0]/2+0.1', degree=3)

for i in [
    'state_100.xml',
    'state_1100.xml',
]:
    print(i)
    target = i.split('.')[0]
    theta = Function(DefaultValues3D.simple_space, 'exp1/' + i)
    theta_n_final = project(NormalDerivativeZ(theta), square)
    theta_n_diff = project(abs(theta_n_final - theta_n), square)
    # to_print = function2d_dumper(
    #     lambda p: abs(theta_n_diff(Point(p[0], p[1], 1))),
    #     folder='scratch', name=target
    # )
    print_2d_isolines(
        theta_n_diff, name=target + '_iso', folder='exp1',
        # table=True,
        # levels=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.05, 0.1]
    )
    print_2d(theta_n_diff, name=target + '_square', folder='exp1', )
    # print_3d_boundaries_on_cube(theta_n_diff, name=target + '_abs', folder='scratch')
