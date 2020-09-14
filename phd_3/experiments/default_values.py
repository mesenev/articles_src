# noinspection PyUnresolvedReferences
from dolfin import (
    FunctionSpace, Function, FiniteElement,
    Expression, FacetNormal, project, VectorFunctionSpace, Constant,
    UnitCubeMesh, grad, dot, BoundaryMesh, UserExpression, TestFunction, MeshFunction, dx, ds
)
from dolfin.cpp.common import DOLFIN_EPS
from dolfin.cpp.mesh import SubDomain


class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        answer = abs(x[0] - 1.0) < DOLFIN_EPS or \
                 abs(x[1] - 1.0) < DOLFIN_EPS or \
                 abs(x[2] - 1.0) < DOLFIN_EPS
        return answer and on_boundary


class NewmanBoundary(SubDomain):
    def inside(self, x, on_boundary):
        answer = x[0] < DOLFIN_EPS or \
                 x[1] < DOLFIN_EPS or \
                 x[2] < DOLFIN_EPS
        return answer and on_boundary



class DefaultValues3D:
    omega = UnitCubeMesh(10, 10, 10)
    sub_domains = MeshFunction("size_t", omega, omega.topology().dim() - 1)
    sub_domains.set_all(0)
    DirichletBoundary().mark(sub_domains, 1)
    NewmanBoundary().mark(sub_domains, 2)
    dss = ds(subdomain_data=sub_domains, domain=omega)
    omega_b = BoundaryMesh(omega, 'exterior')
    finite_element = FiniteElement("Lagrange", omega.ufl_cell(), 2)

    state_space = FunctionSpace(omega, finite_element * finite_element)
    simple_space = FunctionSpace(omega, finite_element)
    boundary_vector_space = VectorFunctionSpace(omega_b, 'Lagrange', 1)
    boundary_simple_space = FunctionSpace(omega_b, 'Lagrange', 1)

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

    def recalculate_r(self):
        self.r = project(
            Expression(
                'alpha * b * gamma * pow(theta_b, 4) + alpha * a * theta_b + gamma * a * theta_n', degree=3,
                a=self.a, alpha=self.alpha, b=self.b, gamma=self.gamma,
                theta_b=self.theta_b, theta_n=self.theta_n
            ),
            self.simple_space)


_n_3d = FacetNormal(DefaultValues3D.omega)


def partial_n_3d(x):
    return dot(_n_3d, grad(x))
