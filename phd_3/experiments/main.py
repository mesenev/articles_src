from dolfin import Expression, Constant
from dolfin.cpp.log import set_log_active
from dolfin.cpp.parameter import parameters

from default_values import DefaultValues3D
from solver import Problem
from utilities import clear_dir, print_3d_boundaries_on_cube

set_log_active(False)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


def make_pics(problem: Problem, name_modifier: str, folder: str):
    # print_3d_boundaries_on_cube(problem.psi_n, name=f'{name_modifier}_control', folder=folder)
    # print_3d_boundaries_single(problem.phi_n, name=f'{name_modifier}_control', folder=folder)
    # print_3d_boundaries_on_cube(problem.state.split()[0], name=f'{name_modifier}_theta', folder=folder)
    # print_3d_boundaries_single(problem.state.split()[0], name=f'{name_modifier}_theta', folder=folder)
    # print_3d_boundaries_on_cube(problem.state.split()[1], name=f'{name_modifier}_phi', folder=folder)
    # print_3d_boundaries_single(problem.state.split()[1], name=f'{name_modifier}_phi', folder=folder)
    # print_3d_boundaries_on_cube(problem.target_diff(), name=f'diff_{name_modifier}_theta', folder=folder)
    return 0


def experiment_1(folder='exp1'):
    clear_dir(folder)
    default_values = DefaultValues3D(
        theta_n=Expression('x[0]/2+0.1', degree=3),
        theta_b=Constant(0.3),
        psi_n_init=Expression('cos(x[1])', degree=3)
    )

    problem = Problem(default_values=default_values)
    answer = problem.solve_boundary()
    print_3d_boundaries_on_cube(answer[0], name=f'theta', folder=folder)
    print_3d_boundaries_on_cube(answer[1], name=f'psi', folder=folder)
    new = Problem(DefaultValues3D(
        theta_n=Expression('x[0]/2+0.1', degree=3),
        theta_b=answer[0],
        psi_n_init=Expression('cos(x[1])', degree=3)
    )).solve_boundary()
    print_3d_boundaries_on_cube(new[0], name=f'theta_new', folder=folder)
    print_3d_boundaries_on_cube(new[1], name=f'psi_new', folder=folder)
    # p = problem.find_optimal_control(iterations=100, _lambda=1)
    # print_3d_boundaries_on_cube(answer[0], name=f'theta_end', folder=folder)
    # print_3d_boundaries_on_cube(answer[1], name=f'psi_end', folder=folder)
    # print_3d_boundaries_on_cube(problem.psi_n, name=f'psi_n', folder=folder)

if __name__ == "__main__":
    experiment_1()
    # experiment_2()
    # experiment_3()