from os import listdir
from os.path import isfile, join

from experiment1 import problem
from matplotlib import tri
from mshr import (Rectangle, Circle, generate_mesh)
from dolfin import *
from matplotlib import pyplot as plt
from utilities import NormalDerivativeZ, print_2d, NormalDerivativeZ_0

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

folder = 'exp1_saved'
set_log_active(False)

omega2d = UnitSquareMesh(50, 50)
square = FunctionSpace(omega2d, FiniteElement("CG", omega2d.ufl_cell(), 1))
xml_files = [f for f in listdir(folder) if isfile(join(folder, f)) and f.split('.')[1] == 'xml']
# omega_circle = generate_mesh(Rectangle(Point(0., 0.), Point(1., 1.)) - Circle(Point(0.5, 0.5), .25), 100)
# f_space = FunctionSpace(omega_circle, FiniteElement("CG", omega_circle.ufl_cell(), 1))
# t = Function(f_space)
triangulation = tri.Triangulation(
    *omega2d.coordinates().reshape((-1, 2)).T,
    triangles=omega2d.cells()
)
for file_name in filter(lambda x: x.split('.')[0] != 'mesh', xml_files):
    theta = Function(problem.theta.function_space(), f'{folder}/{file_name}')
    theta_n = problem.def_values.theta_n
    theta_n_final = project(NormalDerivativeZ(theta, problem.def_values.vector_space), square)
    theta_n_diff = project(abs(theta_n_final - theta_n) / abs(theta_n), square)
    # print_2d(theta_n_diff, name=file_name+'_diff', folder=folder)
    # theta_n_final_0 = project(NormalDerivativeZ_0(theta, problem.def_values.vector_space), square)
    # theta_n_diff = project(abs(theta_n_final_0 - theta_n) / abs(theta_n), square)
    # print_2d(theta_n_diff, name=file_name+'_diff_0', folder=folder)
    z = theta_n_diff.compute_vertex_values(omega2d)


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
