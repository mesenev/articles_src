import os
import shutil

from direct_solve import DirectSolve
from phd_2.experiments.utilities import *
from phd_2.optimization.default_values import ThetaN
from phd_2.optimization.solver import SolveOptimization, SolveBoundary

set_log_active(False)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


def clear_dir(folder):
    try:
        shutil.rmtree(folder)
    except OSError:
        print("Deletion of the directory %s failed" % folder)
    finally:
        os.mkdir(folder)


def make_pics(problem: SolveBoundary, name_modifier: str, folder: str):
    print_3d_boundaries_on_cube(problem.phi_n, name=f'{name_modifier}_control', folder=folder)
    print_3d_boundaries_single(problem.phi_n, name=f'{name_modifier}_control', folder=folder)
    print_3d_boundaries_on_cube(problem.state.split()[0], name=f'{name_modifier}_theta', folder=folder)
    print_3d_boundaries_single(problem.state.split()[0], name=f'{name_modifier}_theta', folder=folder)
    print_3d_boundaries_on_cube(problem.state.split()[1], name=f'{name_modifier}_phi', folder=folder)
    print_3d_boundaries_single(problem.state.split()[1], name=f'{name_modifier}_phi', folder=folder)
    print_3d_boundaries_on_cube(problem.target_diff(), name=f'diff_{name_modifier}_theta', folder=folder)
    return 0


def experiment_1(folder='exp1'):
    clear_dir(folder)

    problem = SolveOptimization()
    r_default = Expression("x[1]", degree=3)
    problem._r = r_default
    problem.phi_n = Expression('0.1 + x[0]*0.5', degree=3)
    problem.solve_boundary()
    make_pics(problem, 'target', folder)
    target_phi_n = Expression('t', degree=3, t=interpolate(problem.phi_n, problem.simple_space))

    print('Setting up optimization problem')
    problem.theta_b = Expression('t', degree=3, t=interpolate(problem.state.split()[0], problem.simple_space))
    problem.phi_n = Constant(0.1)
    answer = problem.solve_boundary().split()
    print('Boundary init problem is set. Working on setting optimization problem.')
    make_pics(problem, 'init', folder)
    print_3d_boundaries_on_cube(
        project((target_phi_n - problem.phi_n) ** 2, problem.boundary_simple_space),
        name='diff_init_control', folder=folder
    )

    print('Launching iterations')
    problem.find_optimal_control(iterations=1 * 10 ** 2, _lambda=5000)
    make_pics(problem, 'end', folder)
    phi_n = Expression('t', degree=3, t=interpolate(problem.phi_n, problem.simple_space))
    print_3d_boundaries_on_cube(
        project((target_phi_n - phi_n) ** 2, problem.boundary_simple_space),
        name='diff_end_control', folder=folder
    )
    draw_simple_graphic(problem.quality_history, 'quality', folder=folder)
    print('ggwp all done!')


def experiment_2(folder='exp2'):
    clear_dir(folder)
    theta_b = Expression('x[2]*0.1+0.3', degree=3)
    theta_n = ThetaN()
    problem = SolveOptimization(theta_b=theta_b, theta_n=theta_n)

    print('Setting up optimization problem')
    problem.solve_boundary()
    print('Boundary init problem is set. Working on setting optimization problem.')
    make_pics(problem=problem, name_modifier='init', folder=folder)

    print('Launching iterations')

    problem.find_optimal_control(iterations=1 * 10 ** 2, _lambda=100)

    make_pics(problem=problem, name_modifier='end', folder=folder)
    draw_simple_graphic(problem.quality_history, 'quality', folder=folder)

    print('ggwp all done!')


def experiment_3(folder='exp3'):
    """
    Currently not working
    :param folder: folder name for output pics
    :return: 0
    """
    clear_dir(folder)

    test = SolveBoundary()
    r_default = Expression("0.2 * (x[0] + x[1] + x[2]) + 0.1", degree=3)
    test._r = r_default
    test.phi_n = Expression('0.5 * (x[0] + x[1])', degree=3)
    test.solve_boundary()
    target_phi_n = Expression('t', degree=3, t=interpolate(test.phi_n, test.simple_space))
    make_pics(test, 'target', folder)

    problem = DirectSolve()
    problem._r = r_default
    problem.theta_b = Expression('t', degree=3, t=interpolate(test.state.split()[0], test.simple_space))
    problem.solve_direct()

    make_pics(problem=problem, name_modifier='end', folder=folder)

    phi_n = Expression('t', degree=3, t=interpolate(problem.state.split()[3], problem.simple_space))
    print_3d_boundaries_on_cube(phi_n, name='end_control', folder=folder)
    print_3d_boundaries_on_cube(
        project((target_phi_n - phi_n) ** 2, problem.boundary_simple_space),
        name='diff_end_control', folder=folder
    )
    print('ggwp all done!')
    return 0


if __name__ == "__main__":
    experiment_1()
    experiment_2()
    # experiment_3()
