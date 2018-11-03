import shutil
import time

from dolfin import Constant

from solvers import SolveReverse
from utilities import print_3d_boundaries_single


def main():
    start_time = time.time()
    # checkers()
    print('Cleaning up a results folder')
    try:
        shutil.rmtree('results/*')
    except:
        pass
    problem = SolveReverse(theta_b=Constant(0.5))
    state = problem.solve_boundary_with_deflection()
    problem.set_theta_0(state[0])
    print(problem.quality())
    problem.solve_boundary_with_phi_n_der()
    print(problem.quality())
    problem.solve_reverse(tolerance=1e-20, iterations=600000)
    problem.get_gamma_from_phi_n_derivative()
    print_3d_boundaries_single(problem.gamma, name='founded_gamma_latest', folder='results')
    end_time = time.gmtime(time.time() - start_time)
    print("Job's done! Elapsed time: {}".format(time.strftime("%H:%M:%S", end_time)))
    return problem


if __name__ == "__main__":
    main()

