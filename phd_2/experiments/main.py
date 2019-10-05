import shutil
import time

from dolfin import Constant

from .solvers_old import SolveReverse
from .utilities import *


def main():
    start_time = time.time()
    # checkers()
    print('Cleaning up a results folder')
    try:
        shutil.rmtree('results/*')
    except:
        pass
    problem = SolveReverse(theta_0=Constant(0.3))
    print('Solving boundary problem with deflection')
    state = problem.solve_boundary_with_phi_n_der()
    print_2d_boundaries(state[0], 'initial_theta', folder='results')
    print_2d_boundaries(state[1], 'initial_phi', folder='results')
    # Set up theta_0 for quality functional
    print(problem.quality())
    # print('Solving boundary problem with phi_n')
    # problem.solve_boundary_with_phi_n_der()
    print(problem.quality())
    print('Solve reverse problem')
    problem.solve_reverse(tolerance=1e-20, iterations=5000)
    print_2d_boundaries(problem.gamma, name='founded_gamma_latest', folder='results')
    end_time = time.gmtime(time.time() - start_time)
    print("Job's done! Elapsed time: {}".format(time.strftime("%H:%M:%S", end_time)))
    return problem


if __name__ == "__main__":
    main()
