# noinspection PyUnresolvedReferences
from dolfin import dx, ds
from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
from dolfin.cpp.parameter import parameters

from solver import Problem
from utilities import print_3d_boundaries_on_cube

set_log_active(False)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


class DirichletBoundary2D(SubDomain):
    def inside(self, x, on_boundary):
        top = abs(x[0] - 1) < DOLFIN_EPS
        bottom = x[0] < DOLFIN_EPS
        right = abs(x[1] - 1) < DOLFIN_EPS
        left = x[1] < DOLFIN_EPS
        answer = top or bottom or right or left
        return answer and on_boundary


class NewmanBoundary2D(SubDomain):
    def inside(self, x, on_boundary, **kwargs):
        answer = 0.1 < x[0] < 0.9 and 0.1 < x[1] < 0.9
        return answer and on_boundary


class DefaultValues2D:
    domain = Rectangle(dolfin.Point(0., 0.), dolfin.Point(1., 1.)) - \
             Circle(dolfin.Point(0.5, 0.5), .2)
    omega = generate_mesh(domain, 100)
    sub_domains = MeshFunction("size_t", omega, omega.topology().dim() - 1)
    sub_domains.set_all(0)
    DirichletBoundary2D().mark(sub_domains, 1)
    NewmanBoundary2D().mark(sub_domains, 2)
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
        self.dirichlet_boundary = DirichletBoundary2D()

    def recalculate_r(self):
        self.r = project(
            Expression(
                'alpha * b * gamma * pow(theta_b, 4) '
                '+ alpha * a * theta_b + gamma * a * theta_n', degree=2,
                a=self.a, alpha=self.alpha, b=self.b, gamma=self.gamma,
                theta_b=self.theta_b, theta_n=self.theta_n
            ),
            self.simple_space)


def experiment_2(folder='exp2'):
    problem.solve_boundary()
    # print_3d_boundaries_on_cube(
    #     problem.theta, name='theta_init', folder='exp2'
    # )
    problem.quality()
    f = File(f'{folder}/solution_0.xml')
    f << problem.theta

    iterator = problem.find_optimal_control(0.2)
    for i in range(10 ** 1):
        next(iterator)
        #
        _diff = problem.quality_history[-2] - problem.quality_history[-1]
        print(f'Iteration {i},\tquality: {problem.quality_history[-1]},\t{_diff}')
        if not i % 100:
            problem.lambda_ += 0.1
        if i in [5, 25, 50, 100, 1000, 5000, 10000]:
            f = File(f'{folder}/solution_{i}.xml')
            f << problem.theta
            print_3d_boundaries_on_cube(
                problem.theta, name=f'theta_{i}', folder='exp2/'
            )
        with open(f'{folder}/quality.txt', 'w') as f:
            print(*problem.quality_history, file=f)


if __name__ == "__main__":
    folder = 'exp2'
    default_values = DefaultValues2D(
        theta_n=Constant(0.2),
        theta_b=Expression('0.2 + x[1] / 2', degree=2),
        psi_n_init=Expression('-0.4 + x[1] / 2', degree=2),
    )

    problem = Problem(default_values=default_values)
    problem.solve_boundary()

    c = plot(problem.theta)
    plt.colorbar(c)
    plt.savefig(f'{folder}/theta_init.png')
    plot(problem.psi)
    plt.colorbar(c)
    plt.savefig(f'{folder}/psi_init.png')
    iterator = problem.find_optimal_control(2)
    next(iterator)
    for i in range(10 ** 3):
        next(iterator)
        _diff = problem.quality_history[-2] - problem.quality_history[-1]
        print(f'Iteration {i},\tquality: {problem.quality_history[-1]},\t{_diff}')

    f = File(f'{folder}/theta_end.xml')
    f << problem.theta
    f = File(f'{folder}/psi_end.xml')
    f << problem.psi
    f = File(f'{folder}/control_end.xml')
    f << problem.psi_n
    with open(f'{folder}/quality.txt', 'w') as f:
        print(*problem.quality_history, file=f)