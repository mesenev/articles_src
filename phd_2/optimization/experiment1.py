from phd_2.optimization.solver import SolveOptimization
from utilities import *

folder = 'exp1'

# clear_dir(folder)
print('Experiment one in a run')

problem = SolveOptimization()
r_default = Constant(0.5)
problem._r = r_default
problem.phi_n = Constant(0.7)
problem.solve_boundary()
print('Setting up optimization problem')
problem.theta_b = Expression('t', degree=3, t=interpolate(problem.state.split()[0], problem.simple_space))
File(f'{folder}/theta_b.xml') << project(problem.theta_b, problem.simple_space)
theta_n = project(get_normal_derivative_3d(problem.state.split()[0]), problem.simple_space)
f = File(f'{folder}/theta_n.xml')
f << theta_n
problem.phi_n = Constant(0.1)
answer = problem.solve_boundary().split()
print('Boundary init problem is set. Working on setting optimization problem.')

print('Launching iterations')
iterator = problem.find_optimal_control(iterations=1 * 10 ** 2, _lambda=1000)

for i in range(9):
    next(iterator)
f = File(f'{folder}/solution_10.xml')
f << problem.state
for i in range(90):
    next(iterator)
f = File(f'{folder}/solution_100.xml')
f << problem.state

for i in range(101, 1001):
    next(iterator)
    if i % 100:
        f = File(f'{folder}/solution_{i}.xml')
        f << problem.state
f = File(f'{folder}/solution_final.xml')
f << problem.state
