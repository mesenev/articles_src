from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import matplotlib.tri as tri
from experiment3 import *
from dolfin import *
from mshr import *
from mshr.cpp import Rectangle, generate_mesh

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

folder = 'exp3'
set_log_active(False)


# class NormalDerivativeZ(UserExpression):
#     def __init__(self, func, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.func = project(grad(func), DefaultValues3D.vector_space)
#
#     def eval(self, value, x):
#         value[0] = self.func(x[0], x[1], 1)[2]
#
#     def __floordiv__(self, other):
#         pass

class FunctionWrapper(UserExpression):
    def __init__(self, func, *args, **kwargs):
        self.ggwp = func
        super().__init__(*args, **kwargs)

    def eval(self, value, x):
        value[0] = self.ggwp(Point(x[0], x[1], 0.5))


domain = Rectangle(dolfin.Point(0., 0.), dolfin.Point(1., 1.)) - \
         Circle(dolfin.Point(0.5, 0.5), .25)
omega_2d = generate_mesh(domain, 100)
finite_element = FiniteElement("CG", omega_2d.ufl_cell(), 1)
f_space = FunctionSpace(omega_2d, finite_element)

xml_files = [f for f in listdir(folder) if isfile(join(folder, f)) and f.split('.')[1] == 'xml']
problem.solve_boundary()
iterator = problem.find_optimal_control(2)
next(iterator), next(iterator), next(iterator)
theta = Function(problem.theta.function_space(), f'{folder}/solution_final.xml')
# z = theta.compute_vertex_values(default_values.omega)
t = Function(f_space)
t.interpolate(FunctionWrapper(theta, element=f_space.ufl_element()))
# theta_projection = interpolate(FunctionWrapper(theta), f_space)
z = t.compute_vertex_values(omega_2d)
# print(z)

triangulation = tri.Triangulation(
    *omega_2d.coordinates().reshape((-1, 2)).T,
    triangles=omega_2d.cells()
)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
plt.tricontour(
    triangulation, z, colors='k', linewidths=0.4,
    levels=list(0.1 + 0.01 * i for i in range(100)), extent=[0, 100, 0, 100]
)
# plt.colorbar()
plt.savefig('plot.svg')
# for i in xml_files:
#     print(i)
#     target = i.split('.')[0]
#     theta = Function(problem.theta.function_space(), f'{folder}/{i}')
# theta_n_final = project(NormalDerivativeZ(theta), square)
# theta_n = problem.def_values.theta_n
# theta_n_diff = project(abs(theta_n_final - theta_n) / abs(theta_n), square)
# to_print = function2d_dumper(
#     lambda p: abs(theta_n_diff(Point(p[0], p[1], 1))),
#     folder='scratch', name=target
# )
# print_2d_isolines(
#     theta_n_diff, name=target + '_iso', folder=folder,
# theta, name=target + '_iso', folder=folder,
# table=True,
# levels=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.05, 0.1]
# )
# print_2d(theta_n_diff, name=target + '_square', folder=folder, )

# with open('exp1/quality.txt', 'r') as f:
#     data = list(map(float, f.read().split()))
# draw_simple_graphic(data, name='quality', folder=folder)
