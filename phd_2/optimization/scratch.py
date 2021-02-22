from dolfin import *

from phd_2.optimization.solver import SolveOptimization
from utilities import print_2d_isolines, get_normal_derivative_3d, print_3d_boundaries_on_cube

problem = SolveOptimization()
r_default = Expression("0.8 * cos(x[0]*3.14/2) + 0.1", degree=3)
answer = Function(SolveOptimization.state_space, 'exp1/solution.xml')
theta_b = Function(SolveOptimization.simple_space, 'exp1/theta_b.xml')
theta_n = Expression('r/a - tb', r=r_default, a=problem.a, tb=theta_b, degree=3)
theta = answer.split()[0]
theta_n_diff = project(abs(project(
    project(get_normal_derivative_3d(theta), problem.simple_space) - theta_n, problem.simple_space
)))
print_3d_boundaries_on_cube(theta_n_diff, name='theta_n_diff_abs', folder='exp1')
