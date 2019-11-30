from dolfin.cpp.log import set_log_active

from phd_2.experiments.utilities import (
    print_2d_boundaries, print_two_with_colorbar,
    draw_simple_graphic
)
from solver import SolveOptimization


def main():
    set_log_active(False)

    problem = SolveOptimization()
    problem.solve_boundary().split()
    init_control = problem.phi_n
    # print_3d_boundaries_single(init_control, name='init_control')
    print_2d_boundaries(init_control, name='init_control', terminal_only=False)
    print_two_with_colorbar(*problem.state.split(), name='init_state')

    control = problem.find_optimal_control(iterations=10 ** 2, _lambda=0.1)
    # print_3d_boundaries_single(control, name='control')
    print_two_with_colorbar(*problem.state.split(), name='state')
    print_2d_boundaries(control, name='control', terminal_only=False)
    draw_simple_graphic(problem.quality_history, 'quality')
    pass


if __name__ == "__main__":
    main()
