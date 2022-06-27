# noinspection PyUnresolvedReferences
from dolfin import dx, ds
# noinspection PyUnresolvedReferences
from matplotlib import tri
# noinspection PyUnresolvedReferences
from mshr import Sphere, Box, generate_mesh, Rectangle, Circle

from phd_3.experiments.consts import DIRICHLET, NEWMAN
from dolfin import *
from dolfin.cpp.parameter import parameters

from os import listdir
from os.path import isfile, join

from phd_3.experiments.default import der
from utilities import print_2d_isolines, print_2d, draw_simple_graphic, print_3d_boundaries_on_cube, clear_dir, \
    NormalDerivativeZ, Wrapper
from phd_3.experiments.meshes.meshgen import CUBE_CIRCLE
from phd_3.experiments.solver import Problem
import matplotlib.pyplot as plt

# set_log_active(False)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


class OutsideCubeBoundary(SubDomain):
    def inside(self, *args, **kwargs):
        x, on_boundary = args[:2]
        sides = list(abs(x[i] - 1) < DOLFIN_EPS for i in range(3)) + list(x[i] < DOLFIN_EPS for i in range(3))
        return any(sides) and on_boundary


class InsideCircleBoundary(SubDomain):
    def inside(self, x, on_boundary, *args, **kwargs):
        # return on_boundary
        answer = 0.1 < x[0] < 0.9 and 0.1 < x[1] < 0.9 and 0.1 < x[2] < 0.9
        return answer and on_boundary


class DefaultValues3D:
    omega = Mesh(CUBE_CIRCLE)
    sub_domains = MeshFunction("size_t", omega, omega.topology().dim() - 1)
    sub_domains.set_all(0)
    a = list()
    OutsideCubeBoundary().mark(sub_domains, DIRICHLET)
    a += [len(list(filter(lambda x: x == 1, sub_domains.array())))]
    InsideCircleBoundary().mark(sub_domains, NEWMAN)
    a.append(len(list(filter(lambda x: x == 1, sub_domains.array()))))
    a.append(len(list(filter(lambda x: x == 2, sub_domains.array()))))
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

    def __init__(self, q_b, theta_b, psi_n_init, **kwargs):
        self.a = 0.6
        self.alpha = 0.333
        self.ka = 1
        self.b = 0.025
        self.beta = 1
        self.gamma = 1
        self.q_b = q_b
        self.psi_n = psi_n_init
        self.theta_b = theta_b
        self.r = None
        self.lmbd = 1
        self.epsilon = 0.1 ** 10
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.recalculate_r()
        self.dirichlet_boundary = OutsideCubeBoundary()

    def recalculate_r(self):
        self.r = project(
            Expression(
                'alpha * b * gamma * pow(theta_b, 4) + alpha * a * theta_b + gamma * a * theta_n', degree=3,
                a=self.a, alpha=self.alpha, b=self.b, gamma=self.gamma,
                theta_b=self.theta_b, theta_n=self.q_b
            ),
            self.simple_space)


class BoundaryExpression(UserExpression):
    def __floordiv__(self, other):
        pass

    def eval(self, value, x):
        value[0] = -0.3
        borders = [x[0], abs(x[0] - 1), x[1], abs(x[1] - 1), x[2], abs(x[2] - 1)]
        if any(map(lambda _: _ < DOLFIN_EPS, borders)):
            value[0] = 0.25


default_values = DefaultValues3D(
    theta_b=Constant(0.1),
    q_b=BoundaryExpression(),
    # q_b=Constant(0.001),
    psi_n_init=Constant(0.5)
)

problem = Problem(default_values=default_values)
problem.solve_boundary()
folder = 'exp3'


def experiment_3():
    problem.quality()
    File(f'{folder}/mesh.xml') << default_values.omega
    File(f'{folder}/solution_0.xml') << problem.theta

    iterator = problem.find_optimal_control(0.2)
    for i in range(10 ** 5 + 1):
        next(iterator)
        #
        _diff = problem.quality_history[-2] - problem.quality_history[-1]
        print(f'Iteration {i},\tquality: {problem.quality_history[-1]},\t{_diff}')
        # if not i % 1000:
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
        print_2d_isolines(theta_n_final, name='deriv_n_theta_iso', folder=folder, )
        print_3d_boundaries_on_cube(theta, folder=folder, name='3d')
        wrapped_theta = Wrapper(lambda x: Point(0.5, x[0], x[1]), theta, element=f_space.ufl_element())
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
        experiment_3()
    except:
        print('Keyboard interruption signal. Wrapping out.')
    finally:
        f = File(f'{folder}/solution_final.xml')
        f << problem.theta
        print_3d_boundaries_on_cube(problem.theta, name=f'theta_final', folder=folder)

    post_prod()
