# noinspection PyUnresolvedReferences
from dolfin import dx, ds
from dolfin import *
import matplotlib.pyplot as plt
from dolfin.cpp.parameter import parameters

from phd_3.experiments.default_values import DefaultValues2D
from solver import Problem
from utilities import print_3d_boundaries_on_cube

set_log_active(False)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


def experiment_2(folder='exp2'):
    problem.solve_boundary()
    # print_3d_boundaries_on_cube(
    #     problem.theta, name='theta_init', folder='exp2'
    # )
    problem.quality()
    f = File(f'{folder}/solution_0.xml')
    f << problem.theta

    iterator = problem.find_optimal_control(0.2)
    for i in range(10 ** 1 + 1):
        next(iterator)
        _diff = problem.quality_history[-2] - problem.quality_history[-1]
        print(f'Iteration {i},\tquality: {problem.quality_history[-1]},\t{_diff}')
        if not i % 100:
            problem.lambda_ += 0.1
        if i in [5, 25, 50, 100, 1000, 5000, 10000]:
            f = File(f'{folder}/solution_{i}.xml')
            f << problem.theta
            print_3d_boundaries_on_cube(
                problem.theta, name=f'theta_{i}', folder='exp2/'
            )
        with open(f'{folder}/quality.txt', 'w') as f:
            print(*problem.quality_history, file=f)


if __name__ == "__main__":
    folder = 'exp2'
    default_values = DefaultValues2D(
        q_b=Constant(0.2),
        theta_b=Expression('0.2 + x[1] / 2', degree=2),
        psi_n_init=Expression('-0.4 + x[1] / 2', degree=2),
    )

    problem = Problem(default_values=default_values)
    problem.solve_boundary()

    c = plot(problem.theta)
    plt.colorbar(c)
    plt.savefig(f'{folder}/theta_init.png')
    plot(problem.psi)
    plt.colorbar(c)
    plt.savefig(f'{folder}/psi_init.png')
    iterator = problem.find_optimal_control(2)
    next(iterator)
    try:
        for i in range(10 ** 3):
            next(iterator)
            _diff = problem.quality_history[-2] - problem.quality_history[-1]
            print(f'Iteration {i},\tquality: {problem.quality_history[-1]},\t{_diff}')
    except Exception as e:
        print('interrupted', e)
    finally:
        pass
    f = File(f'{folder}/theta_end.xml')
    f << problem.theta
    f = File(f'{folder}/psi_end.xml')
    f << problem.psi
    f = File(f'{folder}/control_end.xml')
    f << problem.psi_n
    with open(f'{folder}/quality.txt', 'w') as f:
        print(*problem.quality_history, file=f)


