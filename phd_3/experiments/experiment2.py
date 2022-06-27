# noinspection PyUnresolvedReferences
from dolfin import dx, ds
from dolfin import *
import matplotlib.pyplot as plt
from dolfin.cpp.parameter import parameters

from phd_3.experiments.default_values import DefaultValues2D
from solver import Problem
from utilities import clear_dir
from os import listdir
from os.path import isfile, join

from experiment1 import problem
from matplotlib import tri
from dolfin import *
from matplotlib import pyplot as plt

from phd_3.experiments.default_values import DefaultValues2D
from phd_3.experiments.solver import Problem
from utilities import Wrapper, NormalDerivativeZ, NormalDerivativeZ_0

set_log_active(False)


def fmt(x):
    s = f"{x:.3f}"
    return s


parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
folder = 'exp2'
default_values = DefaultValues2D(
    q_b=Constant(0.2),
    theta_b=Expression('0.2 + x[1] / 2', degree=2),
    psi_n_init=Expression('-0.4 + x[1] / 2', degree=2),
)
problem = Problem(default_values=default_values)
problem.solve_boundary()


def draw_2d_complex(square, omega2d, function, filename):
    t = Function(square)
    t.interpolate(function)
    z = t.compute_vertex_values(omega2d)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    triangulation = tri.Triangulation(
        *omega2d.coordinates().reshape((-1, 2)).T,
        triangles=omega2d.cells()
    )
    cs = ax.tricontour(
        triangulation, z, colors='k', linewidths=0.4,
        extent=[0, 100, 0, 100]
    )
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=7)
    plt.savefig(f'{folder}/{filename}.svg')
    plt.tricontourf(
        triangulation, z, linewidths=0.4,
        # levels=list(0.1 + 0.01 * i for i in range(100)),
        extent=[0, 100, 0, 100]
    )
    plt.colorbar()
    plt.savefig(f'{folder}/{filename}f.svg')



def experiment_2(iterations=20):
    clear_dir(folder)
    c = plot(problem.theta)
    plt.colorbar(c)
    plt.savefig(f'{folder}/theta_init.png')
    plot(problem.psi)
    plt.savefig(f'{folder}/psi_init.png')
    iterator = problem.find_optimal_control(2)
    next(iterator)
    try:
        for _ in range(iterations + 1):
            next(iterator)
            _diff = problem.quality_history[-2] - problem.quality_history[-1]
            print(f'Iteration {_},\tquality: {problem.quality_history[-1]},\t{_diff}')
            with open(f'{folder}/quality.txt', 'w') as f:
                print(*problem.quality_history, file=f)
    except Exception as e:
        print('interrupted', e)
    finally:
        pass
    f = File(f'{folder}/theta_end.xml')
    f << problem.theta
    f = File(f'{folder}/psi_end.xml')
    f << problem.psi
    f = File(f'{folder}/control_end.xml')
    f << problem.psi_n
    with open(f'{folder}/quality.txt', 'w') as f:
        print(*problem.quality_history, file=f)
    return 0


def post_prod():
    omega2d = problem.def_values.omega

    square = problem.def_values.simple_space
    theta = Function(problem.theta.function_space(), f'{folder}/theta_end.xml')
    psi = Function(problem.psi.function_space(), f'{folder}/psi_end.xml')
    phi = project(
        # (psi - default_values.a * theta) / (default_values.alpha * default_values.b),
        (psi - default_values.a * theta) / (default_values.alpha),
        problem.def_values.simple_space
    )

    def draw_2d_complex_(function, filename):
        return draw_2d_complex(square, omega2d, function, filename)

    draw_2d_complex_(theta, 'theta_end')
    draw_2d_complex_(phi, 'phi_end')
    draw_2d_complex_(psi, 'psi_end')


if __name__ == "__main__":
    # experiment_2(iterations=20)
    post_prod()
