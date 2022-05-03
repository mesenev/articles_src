# noinspection PyUnresolvedReferences
from dolfin import dx, ds
# noinspection PyUnresolvedReferences
from matplotlib import tri
from mshr import Sphere, Box, generate_mesh

from phd_3.experiments.consts import DIRICHLET, NEWMAN
from dolfin import *
from dolfin.cpp.log import set_log_active
from dolfin.cpp.parameter import parameters

from os import listdir
from os.path import isfile, join
from utilities import print_2d_isolines, print_2d, draw_simple_graphic, print_3d_boundaries_on_cube, clear_dir, \
    NormalDerivativeZ
from phd_3.experiments.meshes.meshgen import CUBE_CIRCLE
from solver import Problem
import matplotlib.pyplot as plt
from mshr import Rectangle, Circle

set_log_active(False)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


class DirichletBoundary(SubDomain):
    def inside(self, *args, **kwargs):
        x, on_boundary = args[:2]
        sides = list(abs(x[i] - 1) < DOLFIN_EPS for i in range(3)) + list(x[i] < DOLFIN_EPS for i in range(3))
        return any(sides) and on_boundary


class NewmanBoundary(SubDomain):
    def inside(self, *args, **kwargs):
        x, on_boundary = args[:2]
        answer = 0.1 < x[0] < 0.9 and 0.1 < x[1] < 0.9 and 0.1 < x[2] < 0.9
        return answer and on_boundary


class Wrapper(UserExpression):
    def __floordiv__(self, other):
        pass

    point = lambda _: Point(_[0], _[1], 0.5)

    def __init__(self, func, *args, **kwargs):
        self.ggwp = func
        super().__init__(*args, **kwargs)

    def eval(self, value, x):
        value[0] = self.ggwp(self.point(x))


class DefaultValues3D:
    omega = Mesh(CUBE_CIRCLE)
    sub_domains = MeshFunction("size_t", omega, omega.topology().dim() - 1)
    sub_domains.set_all(0)
    DirichletBoundary().mark(sub_domains, DIRICHLET)
    NewmanBoundary().mark(sub_domains, NEWMAN)
    dss = ds(subdomain_data=sub_domains, domain=omega)
    omega_b = BoundaryMesh(omega, 'exterior')
    finite_element = FiniteElement("CG", omega.ufl_cell(), 1)

    state_space = FunctionSpace(omega, finite_element * finite_element)
    simple_space = FunctionSpace(omega, finite_element)
    vector_space = VectorFunctionSpace(omega, 'CG', 1)
    boundary_vector_space = VectorFunctionSpace(omega_b, 'CG', 1)
    boundary_simple_space = FunctionSpace(omega_b, 'CG', 1)

    v = TestFunction(simple_space)
    state = Function(state_space)
    theta, psi = Function(simple_space), Function(simple_space)

    def __init__(self, theta_n, theta_b, psi_n_init, **kwargs):
        self.a = 0.6
        self.alpha = 0.333
        self.ka = 1
        self.b = 0.025
        self.beta = 1
        self.gamma = 1
        self.theta_n = theta_n
        self.psi_n = psi_n_init
        self.theta_b = theta_b
        self.r = None
        self.lmbd = 1
        self.epsilon = 0.1 ** 10
        self.init_control = Constant(0.2)
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.recalculate_r()
        self.dirichlet_boundary = DirichletBoundary()

    def recalculate_r(self):
        self.r = project(
            Expression(
                'alpha * b * gamma * pow(theta_b, 4) + alpha * a * theta_b + gamma * a * theta_n', degree=3,
                a=self.a, alpha=self.alpha, b=self.b, gamma=self.gamma,
                theta_b=self.theta_b, theta_n=self.theta_n
            ),
            self.simple_space)


class BoundaryExpression(UserExpression):
    def __floordiv__(self, other):
        pass

    def eval(self, value, x):
        value[0] = 0.25
        # if x[0] < DOLFIN_EPS or x[1] < DOLFIN_EPS or x[2] < DOLFIN_EPS:
        #     value[0] = 0.75


default_values = DefaultValues3D(
    theta_n=Constant(1),
    theta_b=BoundaryExpression(),
    psi_n_init=Constant(0)
)

problem = Problem(default_values=default_values)
problem.solve_boundary()
folder = 'exp3'


def experiment_3(folder='exp3'):
    # print_3d_boundaries_on_cube(
    #     problem.theta, name='theta_init', folder=folder
    # )
    problem.quality()
    File(f'{folder}/mesh.xml') << default_values.omega
    File(f'{folder}/solution_0.xml') << problem.theta

    iterator = problem.find_optimal_control(4)
    for i in range(10 ** 3 + 1):
        next(iterator)
        #
        _diff = problem.quality_history[-2] - problem.quality_history[-1]
        print(f'Iteration {i},\tquality: {problem.quality_history[-1]},\t{_diff}')
        # if not i % 100:
        #     problem.lambda_ += 0.1
        if i in [50, 100, 1000, 5000, 10000]:
            f = File(f'{folder}/solution_{i}.xml')
            f << problem.theta
            # print_3d_boundaries_on_cube(
            #     problem.theta, name=f'theta_{i}', folder=folder
            # )
        with open(f'{folder}/quality.txt', 'w') as f:
            print(*problem.quality_history, file=f)


def post_prod():
    with open(f'{folder}/quality.txt', 'r') as f:
        data = list(map(float, f.read().split()))
    draw_simple_graphic(data, name='quality', folder=folder)

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
        print(file_name)
        target = file_name.split('.')[0]
        theta = Function(problem.theta.function_space(), f'{folder}/{file_name}')
        theta_n_final = project(NormalDerivativeZ(theta, default_values.vector_space), square)
        theta_n = problem.def_values.theta_n
        theta_n_diff = project(abs(theta_n_final - theta_n) / abs(theta_n), square)
        print_2d_isolines(theta_n_diff, name=target + '_iso', folder=folder, )
        print_2d(theta_n_diff, name=target + '_square', folder=folder, )
        print_3d_boundaries_on_cube(theta, folder=folder, name='3d')
        wrapped_theta = Wrapper(theta, element=f_space.ufl_element())
        for slice in [
            ('x', lambda _: Point(0.5, _[0], _[1])),
            ('y', lambda _: Point(_[0], 0.5, _[1])),
            ('z', lambda _: Point(_[0], _[1], 0.5)),
        ]:
            wrapped_theta.point = slice[1]
            t.interpolate(wrapped_theta)
            z = t.compute_vertex_values(omega_circle)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            plt.tricontour(
                triangulation, z, colors='k', linewidths=0.4,
                extent=[0, 100, 0, 100]
            )

            plt.savefig(f'{folder}/{file_name}_plot{slice[0]}.svg')
            plt.tricontourf(
                triangulation, z, linewidths=0.4,
                # levels=list(0.1 + 0.01 * i for i in range(100)),
                extent=[0, 100, 0, 100]
            )
            plt.colorbar()
            plt.savefig(f'{folder}/{file_name}_plotf{slice[0]}.svg')


if __name__ == "__main__":
    clear_dir(folder)
    try:
        experiment_3(folder)
    except KeyboardInterrupt:
        print('Keyboard interruption signal. Wrapping out.')
    finally:
        f = File(f'{folder}/solution_final.xml')
        f << problem.theta
        print_3d_boundaries_on_cube(problem.theta, name=f'theta_final', folder=folder)

    post_prod()
    # theta = Function(problem.theta.function_space(), f'{folder}/solution_final.xml')
