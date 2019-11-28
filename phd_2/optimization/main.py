from phd_2.experiments.utilities import print_2d_boundaries, print_simple_graphic
from solver import SolveOptimization


def main():
    problem = SolveOptimization()
    # state = problem.solve_boundary().split()
    # print_2d_boundaries(problem.phi_n, terminal_only=True)
    # print('Theta: ')
    # print_2d_boundaries(state[0], terminal_only=True)
    # print('Phi: ')
    # print_2d_boundaries(state[1], terminal_only=True)

    # p1, p2 = problem.solve_conjugate().split()
    control = problem.find_optimal_control(iterations=10 ** 1)
    print_2d_boundaries(control, terminal_only=True)
    print_simple_graphic(problem.quality_history)
    pass


if __name__ == "__main__":
    main()
