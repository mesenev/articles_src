from dolfin import *
from dolfin.cpp.log import set_log_active
from dolfin.cpp.parameter import parameters

from default_values import DefaultValues3D
from solver import Problem
from utilities import clear_dir, print_3d_boundaries_on_cube, get_normal_derivative_3d, print_3d_boundaries_separate, \
    print_2d, Wrapper

set_log_active(False)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


def get_normal_derivative(x):
    return get_normal_derivative_3d(
        x, f_space=DefaultValues3D.simple_space,
        v_space=DefaultValues3D.vector_space
    )


theta_init = Function(DefaultValues3D.simple_space, 'init_vals/theta.xml')
q_b = get_normal_derivative(project(theta_init * 0.6))

phi = Function(DefaultValues3D.simple_space, 'init_vals/phi.xml')
phi_n = get_normal_derivative(phi)
gamma = project(
    (0.333 * phi_n / (phi - theta_init ** 4)),
    DefaultValues3D.simple_space
)

default_values = DefaultValues3D(
    q_b=Constant(0.5),
    theta_b=Expression('0.1 + x[2] / 2', degree=3),
    psi_n_init=Constant(0)
)

problem = Problem(default_values=default_values)
problem.solve_boundary()
folder = 'exp1'

theta = Function(problem.theta.function_space(), f'{folder}/theta_end.xml')
psi = Function(problem.psi.function_space(), f'{folder}/psi_end.xml')
phi = project(
    # (psi - default_values.a * theta) / (default_values.alpha * default_values.b),
    (psi - default_values.a * theta),
    problem.def_values.simple_space
)

# print_3d_boundaries_on_cube(theta, name='theta', folder='results', cmap='viridis', colorbar_scalable=True)
# print_3d_boundaries_on_cube(phi, name='phi', folder='results', cmap='viridis', colorbar_scalable=True)
print_2d(Wrapper(lambda x: [x[0], x[1], 1], theta), 'theta_top', colormap='viridis')
print_2d(Wrapper(lambda x: [x[0], x[1], 1], phi), 'phi_top', colormap='viridis')
