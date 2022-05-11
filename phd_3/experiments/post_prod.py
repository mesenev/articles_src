from os import listdir
from os.path import isfile, join

from experiment1 import problem
from matplotlib import tri
from dolfin import *
from matplotlib import pyplot as plt

from utilities import Wrapper

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

folder = 'exp1'
set_log_active(False)

omega2d = UnitSquareMesh(50, 50)
square = FunctionSpace(omega2d, FiniteElement("CG", omega2d.ufl_cell(), 1))
xml_files = [f for f in listdir(folder) if isfile(join(folder, f)) and f.split('.')[1] == 'xml']
triangulation = tri.Triangulation(
    *omega2d.coordinates().reshape((-1, 2)).T,
    triangles=omega2d.cells()
)
for file_name in filter(lambda x: x.split('.')[0] == 'solution_final', xml_files):
    theta = Function(problem.theta.function_space(), f'{folder}/{file_name}')
    theta_n = problem.def_values.theta_n
    theta_n_final = project(Wrapper(theta, element=square.ufl_element()), square)
    z = theta_n_final.compute_vertex_values(omega2d)


    def fmt(x):
        s = f"{x:.3f}"
        return s

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    cs = ax.tricontour(
        triangulation, z, colors='k', linewidths=0.4,
        extent=[0, 100, 0, 100]
    )

    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=7)
    plt.savefig(f'{folder}/plot{file_name}_contour.svg')
    plt.tricontourf(
        triangulation, z, linewidths=0.4,
        # levels=list(0.1 + 0.01 * i for i in range(100)),
        extent=[0, 100, 0, 100]
    )
    plt.colorbar()
    plt.savefig(f'{folder}/plotf{file_name}_contour.svg')
