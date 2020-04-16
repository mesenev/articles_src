import os
import shutil

from phd_2.experiments.utilities import *
from phd_2.optimization.solver import SolveOptimization


def main():
    set_log_active(False)
    # checkers()
    try:
        shutil.rmtree('results')
    except OSError:
        print("Deletion of the directory %s failed" % 'results')
    finally:
        os.mkdir('results')

    problem = SolveOptimization()
    print('Setting up optimization problem')
    answer = problem.solve_boundary().split()
    print('Boundary init problem is set. Working on setting optimization problem.')
    print_3d_boundaries_single(problem.phi_n, name='init_control')
    print_3d_boundaries_on_cube(answer[0], name='init_theta')
    print_3d_boundaries_single(answer[0], name='init_theta')
    print_3d_boundaries_on_cube(answer[1], name='init_phi')
    print_3d_boundaries_on_cube(problem.target_diff(), name='diff_init_theta')
    print_2d_boundaries(answer[0], name='theta', terminal_only=False)

    print('Launching iterations')
    problem.find_optimal_control(iterations=10 ** 1, _lambda=1000)

    print_3d_boundaries_single(problem.phi_n, name='end_control')
    draw_simple_graphic(problem.quality_history, 'quality')
    print_3d_boundaries_on_cube(problem.target_diff(), name='diff_end_theta')
    print_3d_boundaries_on_cube(problem.state.split()[0], name='end_theta')
    print_3d_boundaries_single(problem.state.split()[0], name='end_theta')
    print_3d_boundaries_on_cube(problem.state.split()[1], name='end_phi')
    print('ggwp all done!')
    return 0


if __name__ == "__main__":
    main()
