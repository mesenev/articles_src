from dolfin import *
from dolfin.cpp.log import set_log_active
from dolfin.cpp.parameter import parameters

from default_values import DefaultValues3D
from solver import Problem
from utilities import clear_dir, print_3d_boundaries_on_cube, get_normal_derivative_3d

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
    q_b=q_b,
    theta_b=Expression('0.1 + x[2] / 2', degree=3),
    psi_n_init=Constant(0)
)

problem = Problem(default_values=default_values)
problem.solve_boundary()
folder = 'exp1'
print_3d_boundaries_on_cube(q_b, name='q_b', folder=folder)
print_3d_boundaries_on_cube(gamma, name='gamma', folder=folder)
print_3d_boundaries_on_cube(theta_init, name='theta_init', folder=folder)
print_3d_boundaries_on_cube(phi, name='phi', folder=folder)
print_3d_boundaries_on_cube(phi_n, name='phi_n', folder=folder)
exit()


def experiment_1():
    print_3d_boundaries_on_cube(
        problem.theta, name='theta_init', folder=folder
    )
    problem.quality()
    File(f'{folder}/solution_0.xml') << problem.theta

    iterator = problem.find_optimal_control(2)
    for i in range(10 ** 3 + 1):
        next(iterator)

        _diff = problem.quality_history[-2] - problem.quality_history[-1]
        print(f'Iteration {i},\tquality: {problem.quality_history[-1]},\t{_diff}')
        if not i % 100:
            problem.lambda_ += 0.1
        if i in [5, 25, 50, 100, 1000, 5000, 10000]:
            File(f'{folder}/solution_{i}.xml') << problem.theta
            File(f'{folder}/control_{i}.xml') << problem.psi_n
        with open(f'{folder}/quality.txt', 'w') as f:
            print(*problem.quality_history, file=f)


if __name__ == "__main__":
    clear_dir(folder)
    try:
        experiment_1()
    except KeyboardInterrupt:
        print('Keyboard interruption signal. Wrapping out.')
    finally:
        File(f'{folder}/solution_final.xml') << problem.theta
        File(f'{folder}/control_final.xml') << problem.psi_n
        with open(f'{folder}/quality.txt', 'w') as f:
            print(*problem.quality_history, file=f)
        print_3d_boundaries_on_cube(problem.theta, name=f'theta_final', folder=folder)
