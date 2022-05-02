# noinspection PyUnresolvedReferences
from dolfin import dx, ds
# noinspection PyUnresolvedReferences
from mshr import Sphere, Box, generate_mesh

from phd_3.experiments.consts import DIRICHLET, NEWMAN
from dolfin import *
from dolfin.cpp.log import set_log_active
from dolfin.cpp.parameter import parameters

from solver import Problem
from utilities import clear_dir, print_3d_boundaries_on_cube

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


class DefaultValues3D:
    domain = Box(Point(0, 0, 0), Point(1, 1, 1)) - \
             Sphere(Point(0.5, 0.5, 0.5), .25)
    omega = generate_mesh(domain, 12)
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
    def eval(self, value, x):
        value[0] = 0.15 if x[1] < 0.5 else 0.85


default_values = DefaultValues3D(
    theta_n=Constant(0.5),
    theta_b=BoundaryExpression(),
    psi_n_init=Constant(0)
)

problem = Problem(default_values=default_values)


def experiment_3(folder='exp3'):
    problem.solve_boundary()
    # print_3d_boundaries_on_cube(
    #     problem.theta, name='theta_init', folder=folder
    # )
    problem.quality()
    f = File(f'{folder}/solution_0.xml')
    f << problem.theta

    iterator = problem.find_optimal_control(2)
    for i in range(10):
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
            break
        with open(f'{folder}/quality.txt', 'w') as f:
            print(*problem.quality_history, file=f)


folder = 'exp3'
if __name__ == "__main__":
    clear_dir(folder)
    try:
        experiment_3(folder)
    except KeyboardInterrupt:
        print('Keyboard interruption signal. Wrapping out.')
    finally:
        f = File(f'{folder}/solution_final.xml')
        f << problem.theta
        # print_3d_boundaries_on_cube(problem.theta, name=f'theta_final', folder=folder)

    theta = Function(problem.theta.function_space(), f'{folder}/solution_final.xml')