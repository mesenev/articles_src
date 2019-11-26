from phd_2.experiments.utilities import print_2d_boundaries
from solver import SolveBoundary


def main():
    problem = SolveBoundary()
    state = problem.solve_boundary().split()
    print('Theta: ')
    print_2d_boundaries(state[0], terminal_only=True)
    print('Phi: ')
    print_2d_boundaries(state[1], terminal_only=True)


if __name__ == "__main__":
    main()
