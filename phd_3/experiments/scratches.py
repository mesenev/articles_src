from pygments.lexer import default

from phd_2.optimization.direct_solve import DirectSolve
from phd_2.optimization.default_values import ThetaN
from phd_2.optimization.solver import SolveOptimization, SolveBoundary
from phd_3.experiments.default_values import DefaultValues3D
from phd_3.experiments.solver import Problem
from utilities import *

set_log_active(False)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

problem = SolveOptimization()


def der(x):
    return get_normal_derivative_3d(x, problem.simple_space, problem.vector_space)


r_default = Expression('0.3 + x[2]/5', degree=3)
problem._r = r_default
problem.phi_n = Expression('0.1 + x[2]/2', degree=3)
problem.solve_boundary()
target_phi_n = Expression('t', degree=3, t=interpolate(problem.phi_n, problem.simple_space))
folder = 'init_vals'
File(f'{folder}/phi_n.xml') << project(problem.phi_n, problem.simple_space)
File(f'{folder}/theta_b.xml') << project(problem.theta_b, problem.simple_space)
File(f'{folder}/r.xml') << project(problem._r, problem.simple_space)
File(f'{folder}/theta.xml') << project(problem.theta, problem.simple_space)
File(f'{folder}/phi.xml') << project(problem.phi, problem.simple_space)

theta_init = Function(problem.simple_space, 'init_vals/theta.xml')
q_b = der(project(theta_init * 0.6))
File(f'{folder}/q_b.xml') << q_b

phi = Function(problem.simple_space, 'init_vals/phi.xml')
phi_n = der(phi)
gamma = project(
    (0.333 * phi_n / (phi - theta_init ** 4)),
    problem.simple_space
)

default_values = DefaultValues3D(
    q_b=q_b,
    theta_b=theta_init,
    psi_n_init=Constant(0.3),
    gamma=gamma,
)

psi_init = project(
    default_values.a * theta_init + default_values.alpha * default_values.b * phi,
    default_values.simple_space
)
print_3d_boundaries_on_cube(psi_init, name='psi_init', folder=folder)


def der(x):
    return get_normal_derivative_3d(x, default_values.simple_space, default_values.vector_space)


#
# problem = Problem(default_values=default_values)
# problem.solve_boundary()
# print_3d_boundaries_on_cube(problem.theta, name='theta_init', folder=folder)
# problem.quality()
# File(f'{folder}/solution_0.xml') << problem.theta
#
# iterator = problem.find_optimal_control(0.1)
# for i in range(101):
#     next(iterator)
#
#     _diff = problem.quality_history[-2] - problem.quality_history[-1]
#     print(f'Iteration {i},\tquality: {problem.quality_history[-1]},\t{_diff}')
#     if not i % 100:
#         problem.lambda_ += 0.1
#     if i in [5, 25, 50, 100, 1000, 5000, 10000]:
#         File(f'{folder}/solution_{i}.xml') << problem.theta
#         File(f'{folder}/control_{i}.xml') << problem.psi_n
#     with open(f'{folder}/quality.txt', 'w') as f:
#         print(*problem.quality_history, file=f)
# File(f'{folder}/theta_final.xml') << problem.theta
# File(f'{folder}/psi_final.xml') << problem.psi
theta = Function(DefaultValues3D.simple_space, f'{folder}/theta_final.xml')
psi = Function(DefaultValues3D.simple_space, f'{folder}/psi_final.xml')
print_3d_boundaries_on_cube(psi, name='psi_final', folder=folder)
# exit(0)
phi = project(
    (psi_init - default_values.a * theta) / (default_values.alpha * default_values.b),
    DefaultValues3D.simple_space
)

print_3d_boundaries_on_cube(phi, name='phi_final', folder=folder)
print_3d_boundaries_on_cube(theta, name='theta_final', folder=folder)
phi_n = der(phi)
print_3d_boundaries_on_cube(phi_n, name='phi_n_final', folder=folder)
gamma_found = project(
    (0.333 * phi_n / (phi - theta ** 4)),
    problem.simple_space
)
gamma_diff = project(abs(gamma - gamma_found), default_values.simple_space)
gamma_diff_proj = Wrapper(lambda x: Point(x[0],  x[1], 1), gamma_diff)
print_2d(gamma_diff_proj, 'gamma_diff_proj', folder=folder)
print_3d_boundaries_on_cube(gamma_found, name='gamma_found', folder=folder)
print_3d_boundaries_on_cube(gamma_diff, name='gamma_diff', folder=folder)
print('hello_world')
