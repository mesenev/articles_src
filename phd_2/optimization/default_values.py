from dolfin import (
    FunctionSpace,
    Function, split, TestFunctions, FiniteElement,
    Expression,
    Constant, DirichletBC, FacetNormal)
from dolfin.cpp.generation import UnitSquareMesh
from dolfin.cpp.mesh import SubDomain
from ufl import grad, dot

theta_n_default = Expression("exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
phi_n_default = Expression("x[0] * sin(x[1])", degree=2)
theta_b = Expression("x[1] * sin(x[0]) + 0.1", degree=2)


# Define Dirichlet boundary
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class DefaultValues:
    omega = UnitSquareMesh(32, 32)
    finite_element = FiniteElement("Lagrange", omega.ufl_cell(), 2)
    state_space = FunctionSpace(omega, finite_element * finite_element)
    simple_space = FunctionSpace(omega, finite_element)
    v, h = TestFunctions(state_space)
    state = Function(state_space)
    theta, phi = split(state)
    theta_bc = DirichletBC(state_space.sub(0), theta_b, Boundary())

    def __init__(self, theta_n=theta_n_default, phi_n=phi_n_default, **kwargs):
        self.a = 0.006
        self.alpha = 0.333
        self.ka = 1
        self.b = 0.025
        self.beta = 1
        self.gamma = 3
        self.theta_n = theta_n
        self.phi_n = phi_n
        for key, val in kwargs:
            setattr(self, key, val)


_n = FacetNormal(DefaultValues.omega)


def partial_n(x):
    return dot(_n, grad(x))
