from dolfin import *
from dolfin.cpp.log import set_log_active
from dolfin.cpp.parameter import parameters

from default_values import DefaultValues3D
from solver import Problem
from utilities import clear_dir, print_3d_boundaries_on_cube

set_log_active(False)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

default_values = DefaultValues3D(
    theta_n=Constant(0.5),
    theta_b=Expression('0.1 + x[2] / 2', degree=3),
    psi_n_init=Constant(0)
)

problem = Problem(default_values=default_values)
problem.solve_boundary()
folder = 'exp1'


def experiment_1():
    print_3d_boundaries_on_cube(
        problem.theta, name='theta_init', folder='exp1'
    )
    problem.quality()
    f = File('exp1/solution_0.xml')
    f << problem.theta

    iterator = problem.find_optimal_control(10)
    for i in range(10 ** 3 + 1):
        next(iterator)
        #
        _diff = problem.quality_history[-2] - problem.quality_history[-1]
        print(f'Iteration {i},\tquality: {problem.quality_history[-1]},\t{_diff}')
        if not i % 100:
            problem.lambda_ += 0.1
        if i in [5, 25, 50, 100, 1000, 5000, 10000]:
            f = File(f'{folder}/solution_{i}.xml')
            f << problem.theta
            print_3d_boundaries_on_cube(
                problem.theta, name=f'theta_{i}', folder='exp1/'
            )
        with open(f'{folder}/quality.txt', 'w') as f:
            print(*problem.quality_history, file=f)


if __name__ == "__main__":
    clear_dir('exp1')
    try:
        experiment_1()
    except KeyboardInterrupt:
        print('Keyboard interruption signal. Wrapping out.')
    finally:
        f = File(f'exp1/solution_final.xml')
        f << problem.theta
        print_3d_boundaries_on_cube(problem.theta, name=f'theta_final', folder='exp1')
