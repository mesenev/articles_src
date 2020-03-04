from phd_2.experiments.utilities import *
from solver import SolveOptimization


def main():
    set_log_active(False)
    # checkers()

    problem = SolveOptimization()
    answer = problem.solve_boundary().split()
    print_3d_boundaries_on_cube(answer[0], name='target_theta')
    print_3d_boundaries_on_cube(answer[1], name='target_phi')
    print_3d_boundaries_single(problem.phi_n, name='target_control')
    print_3d_boundaries_single(answer[0], name='target_theta')
    target_phi_n = Expression(
        't', degree=3, t=interpolate(problem.phi_n, problem.simple_space)
    )
    print(f'\t min: {min(answer[0].vector())},\t max: {max(answer[0].vector())}')

    print('Setting up optimization problem')
    problem.theta_b = Expression(
        't', degree=3, t=interpolate(answer[0], problem.simple_space)
    )
    problem.phi_n = Constant(0.1)
    answer = problem.solve_boundary().split()
    print('Boundary init problem is set. Working on setting optimization problem.')
    print_3d_boundaries_single(problem.phi_n, name='init_control')
    print_3d_boundaries_on_cube(answer[0], name='init_theta')
    print_3d_boundaries_on_cube(answer[1], name='init_phi')
    print_3d_boundaries_on_cube(problem.target_diff(), name='diff_init_theta')
    print_3d_boundaries_on_cube(
        project((target_phi_n - problem.phi_n) ** 2, problem.boundary_simple_space),
        name='diff_init_control'
    )
    # print_2d_boundaries(answer[0], name='theta', terminal_only=False)
    # print_two_with_colorbar(*problem.state.split(), name='init_state')

    print('Launching iterations')
    control = problem.find_optimal_control(iterations=10 ** 1, _lambda=100)
    print_3d_boundaries_single(problem.phi_n, name='end_control')
    # print_two_with_colorbar(*problem.state.split(), name='state')
    # print_2d_boundaries(control, name='control', terminal_only=False)
    draw_simple_graphic(problem.quality_history, 'quality')
    print_3d_boundaries_on_cube(problem.target_diff(), name='diff_end_theta')
    print_3d_boundaries_on_cube(problem.state.split()[0], name='end_theta')
    print_3d_boundaries_on_cube(problem.state.split()[1], name='end_phi')
    phi_n = Expression('t', degree=3, t=interpolate(problem.phi_n, problem.simple_space))
    print_3d_boundaries_on_cube(
        project((target_phi_n - phi_n) ** 2, problem.boundary_simple_space),
        name='diff_end_control'
    )
    print('ggwp all done!')
    return 0


if __name__ == "__main__":
    main()
