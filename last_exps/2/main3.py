# noinspection PyUnresolvedReferences
from dolfin import dx, ds, assemble
from utilities import clear_dir

from matplotlib import tri
from dolfin import *
from matplotlib import pyplot as plt

from phd_3.experiments.default_values import DefaultValues2D
from phd_3.experiments.solver import Problem

set_log_active(False)


def fmt(x):
    s = f"{x:.3f}"
    return s


class ThetaN(UserExpression):
    def __floordiv__(self, other):
        pass

    @staticmethod
    def eval(value, x):
        value[0] = -0.2
        if x[0] < DOLFIN_EPS or abs(x[0] - 1) < DOLFIN_EPS:
            value[0] = 0.2
        if x[1] < DOLFIN_EPS or abs(x[1] - 1) < DOLFIN_EPS:
            value[0] = -0.2


parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

folder = 'exp2'


def solve_optimal(problem, iterations=100):
    problem.solve_boundary()
    iterator = problem.find_optimal_control()
    for _ in range(iterations + 1):
        next(iterator)
        if len(problem.quality_history) > 2:
            _diff = problem.quality_history[-2] - problem.quality_history[-1]
            print(f'Iteration {_},\tquality: {problem.quality_history[-1]},\t{_diff}')
        # with open(f'{folder}/quality.txt', 'w') as f:
        #     print(*problem.quality_history, file=f)
    theta = project(problem.theta, DefaultValues2D.simple_space)
    return theta

result = list()

if __name__ == "__main__":
    q_b_val = project(ThetaN(), DefaultValues2D.simple_space)
    default_values = DefaultValues2D(
        q_b=q_b_val, theta_b=Constant(0.5), psi_n_init=0,
    )

    problem = Problem(default_values=default_values)
    theta_orig = solve_optimal(problem)

    for eps in range(9):
        noise_eps = 0.1 * (-1 + 2 * eps / 8)
        default_values = DefaultValues2D(
            q_b=project(q_b_val + noise_eps, DefaultValues2D.simple_space),
            theta_b=Constant(0.5), psi_n_init=0,
        )
        problem = Problem(default_values=default_values)
        new_theta = solve_optimal(problem)
        scalar = assemble((new_theta - theta_orig) * dx)
        result.append(scalar)
        pass
    with open('result.txt', 'w') as f:
        print(*result, file=f)

# def post_prod():
#
#     omega2d = problem.def_values.omega
#
#     square = problem.def_values.simple_space
#     theta = Function(problem.theta.function_space(), f'{folder}/theta_end.xml')
#     psi = Function(problem.psi.function_space(), f'{folder}/psi_end.xml')
#     phi = project(
#         # (psi - default_values.a * theta) / (default_values.alpha * default_values.b),
#         (psi - default_values.a * theta),
#         problem.def_values.simple_space
#     )
#
#     def draw_2d_complex_(function, filename):
#         return draw_2d_complex(square, omega2d, function, filename)
#
#     draw_2d_complex_(theta, 'theta_end')
#     draw_2d_complex_(phi, 'phi_end')
#     draw_2d_complex_(psi, 'psi_end')
#
