import codecs
import json

from dolfin import *
# noinspection PyUnresolvedReferences
from dolfin import ds, dx
from matplotlib import pyplot as plt, lines
from numpy.ma import arange

from solver2d import SolveOptimization as Problem
from utilities import clear_dir, print_2d_isolines, print_2d_boundaries, draw_simple_graphic, \
    get_normal_derivative

set_log_active(False)

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
print_2d_boundaries(
    get_normal_derivative(theta_0), name='theta_n_0', folder=folder, terminal_only=False
)
print_2d_boundaries(theta_0, name='theta_b_0', folder=folder, terminal_only=False)
print_2d_boundaries(problem.phi_n, name='phi_n_0', folder=folder, terminal_only=False)
print('Boundary init problem is set. Working on setting optimization problem.')

print('Launching iterations')
problem.find_optimal_control(iterations=1 * 10 ** 1, _lambda=20)

theta_100 = problem.state.split()[0]
print_2d_boundaries(
    get_normal_derivative(theta_100), name='theta_n_100', folder=folder, terminal_only=False
)
print_2d_boundaries(theta_100, name='theta_b_100', folder=folder, terminal_only=False)
print_2d_boundaries(problem.phi_n, name='phi_n_100', folder=folder, terminal_only=False)

print_2d_isolines(problem.state.split()[0], name='theta', folder=folder)
draw_simple_graphic(problem.quality_history, name='quality', folder=folder)
draw_simple_graphic(problem.quality_history, name='quality_log', folder=folder, logarithmic=True)
f = File(f'{folder}/state.xml')
f << problem.state
print_2d_boundaries(theta_n, name='theta_n', folder=folder, terminal_only=False)
json.dump(
    problem.quality_history,
    codecs.open(f"{folder}/quality", 'w', encoding='utf-8'),
    separators=(',', ':'), indent=1
)
print('theta_b', assemble(0.5 * (problem.state.split()[0] - init) ** 2 * ds))
print('theta_n', assemble(0.5 * (
        project(get_normal_derivative(theta_100), problem.simple_space) -
        project(get_normal_derivative(init), problem.simple_space)) ** 2 * ds))

figure = plt.figure()
ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
plt.grid(True)
theta_n_original = json.load(open('exp4/theta_n_original', 'r'))
theta_n_0 = json.load(open('exp4/theta_n_0', 'r'))
theta_n_100 = json.load(open('exp4/theta_n_100', 'r'))
ticks_minor = [0, 0.25, 0.5, 0.75, 1]
ticks = [0.125, 0.375, 0.625, 0.875]
ax.set_xticks(ticks)
ax.set_xticks(ticks_minor, minor=True)
x_labels = ['$x_1 = 0$', '$x_2 = 1$', '$x_1 = 1$', '$x_2 = 0$']
ax.set_xticklabels(x_labels)
step = len(theta_n_original['data']) - 1
x = arange(0, 1 + 1 / step, 1 / step)
green_legend = lines.Line2D(
    [], [], color='grey', linestyle='-', label=r'$\partial_n \hat\theta$ заданное значение'
)
blue_legend = lines.Line2D(
    [], [], color='black', linestyle='--', label=r"$\partial_n \theta$ начальное значение"
)
yellow_legend = lines.Line2D(
    [], [], color='black', linestyle='-.', label=r'$\partial_n \theta_\lambda$ приближенное решение'
)
plt.legend(handles=[green_legend, blue_legend, yellow_legend], prop={'size': 10})
plt.plot(x, theta_n_original['data'], "-", color='grey')
plt.plot(x, theta_n_100['data'], '-.', color='black')
plt.plot(x, theta_n_0['data'], '-', color='black')

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
ax.set_xticklabels(x_labels)
step = len(theta_b_original['data']) - 1
x = arange(0, 1 + 1 / step, 1 / step)
green_legend = lines.Line2D(
    [], [], color='grey', linestyle='-', label=r'$\partial_n \hat\theta$ заданное значение'
)
blue_legend = lines.Line2D(
    [], [], color='black', linestyle='--', label=r"$\partial_n \theta$ начальное значение"
)
yellow_legend = lines.Line2D(
    [], [], color='black', linestyle='-.', label=r'$\partial_n \theta_\lambda$ приближенное решение'
)
plt.legend(handles=[green_legend, blue_legend, yellow_legend], prop={'size': 10})
plt.plot(x, theta_b_original['data'], "-", color='grey')
plt.plot(x, theta_b_100['data'], '-.', color='black')
plt.plot(x, theta_b_0['data'], '--', color='black')

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
