from dolfin import *
from dolfin.cpp.log import set_log_active
from dolfin.cpp.parameter import parameters

from default_values import DefaultValues3D
from solver import Problem
from utilities import clear_dir, print_3d_boundaries_on_cube

set_log_active(False)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


class NormalDerivativeZ(UserExpression):
    def __init__(self, func, vs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = project(grad(func), vs)

    def eval(self, value, x):
        value[0] = self.func(x[0], x[1], 1)[-1]


omega2d = UnitSquareMesh(50, 50)
finite_element = FiniteElement("CG", omega2d.ufl_cell(), 1)
square = FunctionSpace(omega2d, finite_element)


def experiment_1(folder='exp1'):
    clear_dir(folder)
    default_values = DefaultValues3D(
        theta_n=Expression('x[0]/2+0.1', degree=3),
        theta_b=Constant(0.3),
        psi_n_init=Expression('cos(x[1])', degree=3)
    )

    problem = Problem(default_values=default_values)
    answer = problem.solve_boundary()
    print_3d_boundaries_on_cube(answer[0], name=f'theta', folder=folder)
    print_3d_boundaries_on_cube(answer[1], name=f'psi', folder=folder)
    new = Problem(DefaultValues3D(
        theta_n=Expression('x[0]/2+0.1', degree=3),
        theta_b=answer[0],
        psi_n_init=Expression('cos(x[1])', degree=3)
    )).solve_boundary()
    print_3d_boundaries_on_cube(new[0], name=f'theta_new', folder=folder)
    print_3d_boundaries_on_cube(new[1], name=f'psi_new', folder=folder)
    problem.find_optimal_control(iterations=100, _lambda=0.01)
    answer = problem.solve_boundary()[0]
    f = File(f'{folder}/state_100.xml')
    f << answer
    problem.find_optimal_control(iterations=10000, _lambda=0.1)
    answer = problem.solve_boundary()[0]
    f = File(f'{folder}/state_1100.xml')
    f << answer
    # print_3d_boundaries_on_cube(answer[0], name=f'theta_end', folder=folder)
    # print_3d_boundaries_on_cube(answer[1], name=f'psi_end', folder=folder)
    # print_3d_boundaries_on_cube(problem.psi_n, name=f'psi_n_end', folder=folder)


if __name__ == "__main__":
    experiment_1()
