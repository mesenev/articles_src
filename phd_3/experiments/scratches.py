from phd_2.optimization.direct_solve import DirectSolve
from phd_2.optimization.default_values import ThetaN
from phd_2.optimization.solver import SolveOptimization, SolveBoundary
from utilities import *

set_log_active(False)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

problem = SolveOptimization()


def der(x):
    return get_normal_derivative_3d(x, problem.simple_space, problem.vector_space)


r_default = Expression('x[1]/2 + 0.1', degree=3)
problem._r = r_default
problem.phi_n = Constant(0.3)
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

phi = Function(problem.simple_space, 'init_vals/phi.xml')
phi_n = der(phi)
gamma = project(
    (0.333 * phi_n / (phi - theta_init ** 4)),
    problem.simple_space
)

print_3d_boundaries_on_cube(q_b, name='q_b', folder=folder)
print_3d_boundaries_on_cube(gamma, name='gamma', folder=folder)
print_3d_boundaries_on_cube(theta_init, name='theta_init_b', folder=folder)
print_3d_boundaries_on_cube(phi, name='phi', folder=folder)
print_3d_boundaries_on_cube(phi_n, name='phi_n', folder=folder)
exit()
