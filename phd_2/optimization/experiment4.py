import codecs
from matplotlib import pyplot as plt, lines
import json

from solver2d import SolveOptimization as Problem
from utilities import clear_dir, print_2d_isolines, print_2d_boundaries, draw_simple_graphic, \
    get_normal_derivative
from dolfin import *
from numpy.ma import arange

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


figure = plt.figure()
ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
# plt.grid(True)
theta_n_original = json.load(open('exp4/theta_n_original', 'r'))
theta_n_0 = json.load(open('exp4/theta_n_0', 'r'))
theta_n_100 = json.load(open('exp4/theta_n_100', 'r'))
ticks_minor = [0, 0.25, 0.5, 0.75, 1]
ticks = [0.125, 0.375, 0.625, 0.875]
ax.set_xticks(ticks)
ax.set_xticks(ticks_minor, minor=True)
x_labels = ['x=0', 'y=1', 'x=1', 'y=0']
ax.set_xticklabels(x_labels)
step = len(theta_n_original['data']) - 1
x = arange(0, 1 + 1 / step, 1 / step)
green_legend = lines.Line2D([], [], color='red', linestyle='-.', label=r'$\partial_n\hat\theta$ эталонное значение')
blue_legend = lines.Line2D([], [], color='blue', linestyle='-', label=r"$\partial_n\theta$ начальное значение")
yellow_legend = lines.Line2D([], [], color='green', linestyle='--', label=r'$\partial_n\theta$ на сотой итерации')
plt.legend(handles=[green_legend, blue_legend, yellow_legend], prop={'size': 10})
plt.plot(x, theta_n_original['data'], "r-.")
plt.plot(x, theta_n_100['data'], 'g--')
plt.plot(x, theta_n_0['data'], 'b-')

ax.grid(True, axis='x', which='minor')
ax.grid(True, axis='y')
# va = [0, -.05, 0, -.05, -.05, -.05]
# for t, y in zip(ax.get_xticklabels(), va):
#     t.set_y(y)
ax.tick_params(axis='x', which='minor', direction='out', length=5)
ax.tick_params(axis='x', which='major', bottom=False, top=False)
plt.xlabel('Граница области')
plt.ylabel('Значение')
plt.savefig(f'{folder}/theta_n.png')


figure = plt.figure()
ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
# plt.grid(True)
theta_b_original = json.load(open('exp4/theta_b_original', 'r'))
theta_b_0 = json.load(open('exp4/theta_b_0', 'r'))
theta_b_100 = json.load(open('exp4/theta_b_100', 'r'))
ticks_minor = [0, 0.25, 0.5, 0.75, 1]
ticks = [0.125, 0.375, 0.625, 0.875]
ax.set_xticks(ticks)
ax.set_xticks(ticks_minor, minor=True)
x_labels = ['x=0', 'y=1', 'x=1', 'y=0']
ax.set_xticklabels(x_labels)
step = len(theta_b_original['data']) - 1
x = arange(0, 1 + 1 / step, 1 / step)
green_legend = lines.Line2D([], [], color='red', linestyle='-.', label=r'$\hat\theta|_\Gamma$ эталонное значение')
blue_legend = lines.Line2D([], [], color='blue', linestyle='-', label=r"$\theta|_\Gamma$ начальное значение")
yellow_legend = lines.Line2D([], [], color='green', linestyle='--', label=r'$\theta|_\Gamma$ на сотой итерации')
plt.legend(handles=[green_legend, blue_legend, yellow_legend], prop={'size': 10})
plt.plot(x, theta_b_original['data'], "r-.")
plt.plot(x, theta_b_100['data'], 'g--')
plt.plot(x, theta_b_0['data'], 'b-')

ax.grid(True, axis='x', which='minor')
ax.grid(True, axis='y')
# va = [0, -.05, 0, -.05, -.05, -.05]
# for t, y in zip(ax.get_xticklabels(), va):
#     t.set_y(y)
ax.tick_params(axis='x', which='minor', direction='out', length=5)
ax.tick_params(axis='x', which='major', bottom=False, top=False)
plt.xlabel('Граница области')
plt.ylabel('Значение')
plt.savefig(f'{folder}/theta_b.png')
