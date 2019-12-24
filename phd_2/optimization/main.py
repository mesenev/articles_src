from dolfin import *
from dolfin.cpp.log import set_log_active

from .solver import SolveOptimization
from ..experiments.utilities import *


def main():
    set_log_active(False)
    # checkers()

    problem = SolveOptimization()
    answer = problem.solve_boundary().split()
    print(answer[0](Point(0.5, 0.5, 0.5)))
    print(answer[1](Point(0.5, 0.5, 0.5)))
    print_3d_boundaries_on_cube(answer[0], name='theta_init')
    print_3d_boundaries_on_cube(answer[1], name='phi')
    print_3d_boundaries_on_cube(problem.target_diff(), name='diff_init')
    # print_2d_boundaries(answer[0], name='theta', terminal_only=False)
    # print_two_with_colorbar(*problem.state.split(), name='init_state')

    control = problem.find_optimal_control(iterations=10 ** 3, _lambda=1)
    print_3d_boundaries_single(control, name='control')
    # print_two_with_colorbar(*problem.state.split(), name='state')
    # print_2d_boundaries(control, name='control', terminal_only=False)
    draw_simple_graphic(problem.quality_history, 'quality')
    print_3d_boundaries_on_cube(problem.target_diff(), name='diff_end')
    pass


if __name__ == "__main__":
    main()
