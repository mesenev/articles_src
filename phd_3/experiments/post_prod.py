from os import listdir
from os.path import isfile, join

from experiment1 import problem
from matplotlib import tri
from dolfin import *
from matplotlib import pyplot as plt

from utilities import Wrapper, NormalDerivativeZ, NormalDerivativeZ_0

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["linear_algebra_backend"] = 'mumps'


def fmt(x):
    s = f"{x:.3f}"
    return s


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
    # theta_n = problem.def_values.theta_n / problem.def_values.a
    # theta_z = project(abs(NormalDerivativeZ(theta, problem.def_values.vector_space) - theta_n), square)
    # theta_z0 = project(abs(NormalDerivativeZ_0(theta, problem.def_values.vector_space) - theta_n), square)

    # for name, t in [('up', theta_z), ('down', theta_z0)]:
    #     z = t.compute_vertex_values(omega2d)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.set_aspect('equal')
    #
    #     cs = ax.tricontour(
    #         triangulation, z, colors='k', linewidths=0.4,
    #         extent=[0, 100, 0, 100]
    #     )
    #
    #     ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=7)
    #     plt.savefig(f'{folder}/plot{name}_contour.svg')
    #     plt.tricontourf(
    #         triangulation, z, linewidths=0.4,
            # levels=list(0.1 + 0.01 * i for i in range(100)),
            # extent=[0, 100, 0, 100]
        # )
        # plt.colorbar()
        # plt.savefig(f'{folder}/plotf{name}_contour.svg')
filename = 'solution_final.xml'
theta = Function(problem.theta.function_space(), f'{folder}/{filename}')
for slice in [
    ('x', lambda _: Point(0.5, _[0], _[1])),
    ('y', lambda _: Point(_[1], 0.5, _[0])),
    ('z', lambda _: Point(_[0], _[1], 0.5)),
]:
    wrapped_theta = Wrapper(slice[1], theta, element=square.ufl_element())
    t = Function(square)
    t.interpolate(wrapped_theta)
    z = t.compute_vertex_values(omega2d)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    cs = ax.tricontour(
        triangulation, z, colors='k', linewidths=0.4,
        extent=[0, 100, 0, 100]
    )
    ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=7)
    plt.savefig(f'{folder}/{filename}_plot{slice[0]}.svg')
    plt.tricontourf(
        triangulation, z, linewidths=0.4,
        # levels=list(0.1 + 0.01 * i for i in range(100)),
        extent=[0, 100, 0, 100]
    )
    plt.colorbar()
    plt.savefig(f'{folder}/{filename}_plotf{slice[0]}.svg')




# noinspection PyUnresolvedReferences

from dolfin import dx, ds
from matplotlib.patches import Circle as pltCircle
# noinspection PyUnresolvedReferences
from matplotlib import tri
from mshr import Sphere, Box, generate_mesh

from consts import DIRICHLET, NEWMAN
from dolfin import *
from dolfin.cpp.log import set_log_active
from dolfin.cpp.parameter import parameters

from os import listdir
from os.path import isfile, join

from phd_3.experiments.experiment3 import DefaultValues3D, BoundaryExpression, Wrapper
from utilities import print_2d_isolines, print_2d, draw_simple_graphic, print_3d_boundaries_on_cube, clear_dir, \
    NormalDerivativeZ
from meshes.meshgen import CUBE_CIRCLE
from solver import Problem
import matplotlib.pyplot as plt
from mshr import Rectangle, Circle

folder = 'exp1'
default_values = DefaultValues3D(
    theta_n=Constant(1),
    theta_b=BoundaryExpression(),
    psi_n_init=Constant(0)
)

problem = Problem(default_values=default_values)
problem.solve_boundary()


def post_prod():
    omega2d = UnitSquareMesh(50, 50)
    square = FunctionSpace(omega2d, FiniteElement("CG", omega2d.ufl_cell(), 1))
    xml_files = [f for f in listdir(folder) if isfile(join(folder, f)) and f.split('.')[1] == 'xml']
    omega_circle = generate_mesh(Rectangle(Point(0., 0.), Point(1., 1.)) - Circle(Point(0.5, 0.5), .25), 100)
    f_space = FunctionSpace(omega_circle, FiniteElement("CG", omega_circle.ufl_cell(), 1))
    t = Function(f_space)
    triangulation = tri.Triangulation(
        *omega_circle.coordinates().reshape((-1, 2)).T,
        triangles=omega_circle.cells()
    )
    file_name = 'solution_final.xml'
    target = file_name.split('.')[0]
    theta = Function(problem.theta.function_space(), f'{folder}/{file_name}')
    # theta_n_final = project(NormalDerivativeZ(theta, default_values.vector_space), square)
    # theta_n = problem.def_values.theta_n
    # theta_n_diff = project(abs(theta_n_final - theta_n) / abs(theta_n), square)
    # print_2d_isolines(theta_n_diff, name=target + '_iso', folder=folder, )
    # print_2d(theta_n_diff, name=target + '_square', folder=folder, )
    # print_3d_boundaries_on_cube(theta, folder=folder, name='3d')

    for slice in [
        ('x', lambda _: Point(0.5, _[0], _[1])),
        ('y', lambda _: Point(_[1], 0.5, _[0])),
        ('z', lambda _: Point(_[0], _[1], 0.5)),
    ]:
        wrapped_theta = Wrapper(slice[1], theta, element=f_space.ufl_element())
        wrapped_theta.point = slice[1]
        t.interpolate(wrapped_theta)
        z = t.compute_vertex_values(omega_circle)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')

        def fmt(x):
            s = f"{x:.3f}"
            return s

        cs = ax.tricontour(
            triangulation, z, colors='k', linewidths=0.4,
            extent=[0, 100, 0, 100]
        )
        ax.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=7)
        ax.add_patch(pltCircle((0.5, 0.5), 0.25, edgecolor='black',
                 facecolor='white', linewidth=0.5))
        plt.savefig(f'{folder}/{file_name}_plot{slice[0]}.svg')
        plt.tricontourf(
            triangulation, z, linewidths=0.4,
            # levels=list(0.1 + 0.01 * i for i in range(100)),
            extent=[0, 100, 0, 100]
        )
        plt.colorbar()
        plt.savefig(f'{folder}/{file_name}_plotf{slice[0]}.svg')

post_prod()
