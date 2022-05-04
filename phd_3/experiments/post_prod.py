from os import listdir
from os.path import isfile, join

from experiment1 import problem
from matplotlib import tri
from mshr import (Rectangle, Circle, generate_mesh)
from dolfin import *

from utilities import NormalDerivativeZ, print_2d

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

folder = 'exp1'
set_log_active(False)

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
for file_name in filter(lambda x: x.split('.')[0] != 'mesh', xml_files):
    theta = Function(problem.theta.function_space(), f'{folder}/{file_name}')
    theta_n_final = project(NormalDerivativeZ(theta, problem.def_values.vector_space), square)
    theta_n = problem.def_values.theta_n
    theta_n_diff = project(abs(theta_n_final - theta_n) / abs(theta_n), square)
    print_2d(theta_n_diff, name=file_name+'_diff', folder=folder)
