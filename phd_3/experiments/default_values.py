# noinspection PyUnresolvedReferences
from abc import ABC

from dolfin import (
    FunctionSpace, Function, FiniteElement,
    Expression, FacetNormal, project, VectorFunctionSpace, Constant,
    UnitCubeMesh, grad, dot, BoundaryMesh, UserExpression, TestFunction, MeshFunction
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


class ThetaN(UserExpression):
    def eval(self, value, x):
        value[0] = 0
        if x[2] + DOLFIN_EPS > 1:
            value[0] = 0.11
        if x[2] - DOLFIN_EPS < 0:
            value[0] = -0.15


theta_n_default_3d = ThetaN()  # Expression("(x[0] + x[1] + x[2])/4 + 0.1", degree=3)
psi_n_default_3d = Constant(0.1)  # Expression("x[0] / 3 * sin(x[1]) + 0.1 + x[2] / 2", degree=3)
theta_b_3d = Expression('x[2]*0.1+0.3', degree=3)


class DefaultValues3D:
    omega = UnitCubeMesh(8, 8, 8)
    sub_domains = MeshFunction("size_t", omega, omega.topology().dim() - 1)
    NewmanBoundary().mark(sub_domains, 0)
    DirichletBoundary().mark(sub_domains, 1)
    omega_b = BoundaryMesh(omega, 'exterior')
    finite_element = FiniteElement("Lagrange", omega.ufl_cell(), 2)

    state_space = FunctionSpace(omega, finite_element * finite_element)
    simple_space = FunctionSpace(omega, finite_element)
    boundary_vector_space = VectorFunctionSpace(omega_b, 'Lagrange', 1)
    boundary_simple_space = FunctionSpace(omega_b, 'Lagrange', 1)

    v = TestFunction(simple_space)
    state = Function(state_space)
    theta, psi = Function(simple_space), Function(simple_space)

    def __init__(self, theta_n=theta_n_default_3d, psi_n=psi_n_default_3d, **kwargs):
        self.a = 0.6
        self.alpha = 0.333
        self.ka = 1
        self.b = 0.025
        self.beta = 1
        self.theta_n = theta_n
        self.psi_n = psi_n
        self.theta_b = theta_b_3d  # Warning! Might be ambiguous
        self.r = None
        self.lmbd = 1
        self.epsilon = 0.1 ** 10
        self.init_control = Constant(0.2)
        self.recalculate_r()
        for key, val in kwargs.items():
            setattr(self, key, val)

    def recalculate_r(self):
        self.r = project(
            Expression(
                'a * (theta_n + theta_b)',
                degree=3,
                a=self.a, theta_n=self.theta_n,
                beta=self.beta, theta_b=self.theta_b
            ),
            self.simple_space)


_n_3d = FacetNormal(DefaultValues3D.omega)


def partial_n_3d(x):
    return dot(_n_3d, grad(x))
