from dolfin import *

from phd_2.optimization.solver import SolveOptimization
from utilities import print_2d_isolines, get_normal_derivative_3d, print_3d_boundaries_on_cube, function2d_dumper

r_default = Constant(0.5)
theta = Function(SolveOptimization.state_space, 'exp1/solution.xml').split()[0]
theta_b = Function(SolveOptimization.simple_space, 'exp1/theta_b.xml')
theta_n = Function(SolveOptimization.simple_space, 'exp1/theta_n.xml')
theta_n_final = project(
    get_normal_derivative_3d(theta),
    SolveOptimization.simple_space
)
theta_n_diff = project(
    abs(theta_n_final - theta_n) / abs(theta_n),
    SolveOptimization.simple_space
)
to_print = function2d_dumper(
    lambda p: abs(theta_n_diff(Point(p[0], p[1], 1))),
    folder='scratch', name='theta_n_diff'
)
print_2d_isolines(
    to_print, name='theta_n_diff_iso', folder='scratch', table=True,
    levels=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.05, 0.1]
)
print_3d_boundaries_on_cube(theta_n_diff, name='theta_n_diff_abs', folder='scratch')
