import codecs
import json
from solver2d import SolveOptimization as Problem

from utilities import clear_dir, Normal, print_2d_isolines, print_2d_boundaries, draw_simple_graphic, \
    get_normal_derivative
from dolfin import *

folder = 'exp4'

clear_dir(folder)
print('Experiment four in a run')
init = project(
    Expression('pow((x[0]-0.5), 2) - 0.5*x[1] + 0.75',
               element=Problem.simple_space.ufl_element()),
    Problem.simple_space
)
theta_n = get_normal_derivative(init)
print_2d_boundaries(theta_n, name='theta_n_original', folder=folder, terminal_only=False)
print_2d_boundaries(init, name='theta_b_original', folder=folder, terminal_only=False)
problem = Problem(
    phi_n=Constant(0),
    theta_n=interpolate(theta_n, Problem.simple_space),
    theta_b=init,
)

print('Setting up optimization problem')
problem.solve_boundary()
theta_0 = problem.state.split()[0]
print_2d_boundaries(get_normal_derivative(theta_0), name='theta_n_0', folder=folder, terminal_only=False)
print_2d_boundaries(theta_0, name='theta_b_0', folder=folder, terminal_only=False)
print_2d_boundaries(problem.phi_n, name='phi_n_0', folder=folder, terminal_only=False)
print('Boundary init problem is set. Working on setting optimization problem.')

print('Launching iterations')
problem.find_optimal_control(iterations=1 * 10 ** 2, _lambda=20)

theta_100 = problem.state.split()[0]
print_2d_boundaries(get_normal_derivative(theta_100), name='theta_n_100', folder=folder, terminal_only=False)
print_2d_boundaries(theta_100, name='theta_b_100', folder=folder, terminal_only=False)
print_2d_boundaries(problem.phi_n, name='phi_n_100', folder=folder, terminal_only=False)

print_2d_isolines(problem.state.split()[0], name='theta', folder=folder)
draw_simple_graphic(problem.quality_history, name='quality', folder=folder)
draw_simple_graphic(problem.quality_history, name='quality_log', folder=folder, logarithmic=True)
f = File(f'{folder}/state.pvd')
f << problem.state
print_2d_boundaries(theta_n, name='theta_n', folder=folder, terminal_only=False)
json.dump(
    problem.quality_history,
    codecs.open(f"{folder}/quality", 'w', encoding='utf-8'),
    separators=(',', ':'), indent=1
)
