
from dolfin import *

from phd_2.optimization.solver import SolveOptimization
from utilities import print_2d_isolines, get_normal_derivative_3d, print_3d_boundaries_on_cube, function2d_dumper, \
    print_2d, NormalDerivativeZ

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
set_log_active(False)
omega2d = UnitSquareMesh(50, 50)
finite_element = FiniteElement("CG", omega2d.ufl_cell(), 1)
square = FunctionSpace(omega2d, finite_element)
problem = SolveOptimization()

problem._r = Constant(0.5)
problem.phi_n = Constant(0.7)
problem.solve_boundary()
theta_n = project(NormalDerivativeZ(problem.state.split()[2]), square)
# exit(0)

for i in [
    'solution_25.xml',
    # 'solution_50.xml',
    # 'solution_75.xml',
    # 'solution_100.xml'
]:
    print(i)
    target = i.split('.')[0]
    theta = Function(SolveOptimization.state_space, 'exp1/' + i).split()[0]
    theta_n_final = project(NormalDerivativeZ(theta), square)
    theta_n_diff = project(abs(theta_n_final - theta_n) / abs(theta_n), square)
    # to_print = function2d_dumper(
    #     lambda p: abs(theta_n_diff(Point(p[0], p[1], 1))),
    #     folder='scratch', name=target
    # )
    print_2d_isolines(
        theta_n_diff, name=target + '_iso', folder='scratch',
        # table=True,
        # levels=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.05, 0.1]
    )
    print_2d(theta_n_diff, name=target + '_square', folder='scratch', )
    # print_3d_boundaries_on_cube(theta_n_diff, name=target + '_abs', folder='scratch')
