from phd_2.optimization.solver import SolveOptimization, SolveBoundary
from utilities import *

folder = 'exp2'

clear_dir(folder)
print('Experiment one in a run')

problem = SolveOptimization()
r_default = Expression("0.8 * cos(x[1]*3.14/2) + 0.5", degree=3)
problem._r = r_default
problem.phi_n = Constant(0.5)  # Expression("0.8 * cos(x[1]*3.14/2) + 0.5", degree=3)
problem.solve_boundary()
print('Setting up optimization problem')
problem.theta_b = Expression('t', degree=3, t=interpolate(problem.state.split()[0], problem.simple_space))
File(f'{folder}/theta_b.xml') << project(problem.theta_b, problem.simple_space)
problem.phi_n = Constant(0)
print('Boundary init problem is set. Working on setting optimization problem.')

print('Launching iterations')
problem.find_optimal_control(iterations=1 * 10 ** 1, _lambda=10 * 9)

theta_n = Expression('r/a - tb', r=r_default, a=problem.a, tb=problem.theta_b, degree=3)
print_3d_boundaries_on_cube(theta_n, name='theta_n', folder=folder)
theta = problem.state.split()[0]
theta_n_diff = project(project(
    get_normal_derivative_3d(theta), problem.simple_space) - theta_n,
    problem.simple_space
)

print_3d_boundaries_on_cube(theta_n_diff, name='theta_n_diff_abs', folder='exp1')
to_print = function2d_dumper(
    lambda p: abs(theta_n_diff(Point(p[0], p[1], 1))),
    folder=folder, name='theta_n_diff'
)
print_2d_isolines(to_print, folder=folder, table=True)
print_3d_boundaries_on_cube(theta_n_diff, name='theta_n_diff', folder=folder)
print_3d_boundaries_on_cube(problem.phi_n, name='phi_n_final', folder=folder)
print_2d(to_print, folder=folder, name='theta_n_diff', table=True)
draw_simple_graphic(problem.quality_history, 'quality', folder=folder)
f = File(f'{folder}/solution.xml')
f << problem.state
print('ggwp all done!')
