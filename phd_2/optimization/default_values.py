# noinspection PyUnresolvedReferences
from abc import ABC

from dolfin import (
    FunctionSpace, Function, split, TestFunctions, FiniteElement,
    Expression, FacetNormal, project, VectorFunctionSpace, Constant,
    UnitCubeMesh, grad, dot, BoundaryMesh, UserExpression, interpolate
)
from dolfin.cpp.common import DOLFIN_EPS


class ThetaN(UserExpression):
    def __floordiv__(self, other):
        pass

    @staticmethod
    def eval(value, x):
        value[0] = 0
        if x[2] + DOLFIN_EPS > 1:
            value[0] = 0.11
        if x[2] - DOLFIN_EPS < 0:
            value[0] = -0.15


theta_n_default_3d = ThetaN()  # Expression("(x[0] + x[1] + x[2])/4 + 0.1", degree=3)
phi_n_default_3d = Constant(0.1)  # Expression("x[0] / 3 * sin(x[1]) + 0.1 + x[2] / 2", degree=3)
theta_b_3d = Expression('x[2]*0.1+0.3', degree=3)


class DefaultValues3D:
    omega = UnitCubeMesh(25, 25, 25)
    omega_b = BoundaryMesh(omega, 'exterior')
    finite_element = FiniteElement("CG", omega.ufl_cell(), 1)

    state_space = FunctionSpace(omega, finite_element * finite_element)
    simple_space = FunctionSpace(omega, finite_element)
    boundary_vector_space = VectorFunctionSpace(omega_b, 'CG', 1)
    vector_space = VectorFunctionSpace(omega, 'CG', 1)
    boundary_simple_space = FunctionSpace(omega_b, 'CG', 1)

    v, h = TestFunctions(state_space)
    state = Function(state_space)
    theta, phi = split(state)

    def __init__(self, theta_n=theta_n_default_3d, phi_n=phi_n_default_3d, **kwargs):
        self.a = 0.6
        self.alpha = 0.333
        self.ka = 1
        self.b = 0.025
        self.beta = 1
        self.theta_n = theta_n
        self.phi_n = phi_n
        self.theta_b = theta_b_3d  # Warning! Might be ambiguous
        self._r = None
        self.recalculate_r()
        for key, val in kwargs.items():
            setattr(self, key, val)

    def recalculate_r(self):
        # self._r = interpolate(
        #     Expression(
        #         'a * (theta_n + theta_b)',
        #         degree=3,
        #         a=self.a, theta_n=self.theta_n,
        #         beta=self.beta, theta_b=self.theta_b
        #     ),
        #     self.simple_space)
        pass


_n_3d = FacetNormal(DefaultValues3D.omega)


def partial_n_3d(x):
    return dot(_n_3d, grad(x))
