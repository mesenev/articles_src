# noinspection PyUnresolvedReferences
from dolfin import (
    FunctionSpace,
    Function, split, TestFunctions, FiniteElement,
    Expression, DirichletBC, FacetNormal, project, VectorFunctionSpace, Constant,
    UnitSquareMesh, UnitCubeMesh, grad, dot, SubDomain, BoundaryMesh
)

theta_n_default_2d = Expression("exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
theta_n_default_3d = Expression("(x[0] + x[1] + x[2])/7 + 0.1", degree=3)
phi_n_default_2d = Constant(0.2)  # Expression("x[0] * sin(x[1])", degree=2)
phi_n_default_3d = Expression("x[0] / 6 * sin(x[1]) + 0.1", degree=2)
theta_b_2d = Expression("x[1] * sin(x[0]) + 0.1", degree=2)
theta_b_3d = Expression("x[0] * sin(x[1] + x[2])/4 + 0.1", degree=3)


# Define Dirichlet boundary
class Boundary(SubDomain):
    # noinspection PyMethodOverridin
    def inside(self, x, on_boundary):
        return on_boundary


class DefaultValues2D:
    omega = UnitSquareMesh(32, 32)
    omega_b = BoundaryMesh(omega, 'exterior')
    finite_element = FiniteElement("Lagrange", omega.ufl_cell(), 2)

    state_space = FunctionSpace(omega, finite_element * finite_element)
    simple_space = FunctionSpace(omega, finite_element)
    boundary_vector_space = VectorFunctionSpace(omega_b, 'Lagrange', 1)

    v, h = TestFunctions(state_space)
    state = Function(state_space)
    theta, phi = split(state)
    theta_bc = DirichletBC(state_space.sub(0), theta_b_2d, Boundary())

    def __init__(self, theta_n=theta_n_default_2d, phi_n=phi_n_default_2d, **kwargs):
        self.a = 0.006
        self.alpha = 0.333
        self.ka = 1
        self.b = 0.025
        self.beta = 1
        self.gamma = 3
        self.theta_n = theta_n
        self.phi_n = phi_n
        self.theta_b = theta_b_2d  # Warning! Might be ambiguous
        self._r = project(
            Expression(
                'theta_n + theta_b',
                element=self.finite_element,
                a=self.a, theta_n=self.theta_n,
                beta=self.beta, theta_b=self.theta_b
            ),
            self.simple_space)
        for key, val in kwargs:
            setattr(self, key, val)


class DefaultValues3D:
    omega = UnitCubeMesh(8, 8, 8)
    omega_b = BoundaryMesh(omega, 'exterior')
    finite_element = FiniteElement("Lagrange", omega.ufl_cell(), 2)

    state_space = FunctionSpace(omega, finite_element * finite_element)
    simple_space = FunctionSpace(omega, finite_element)
    boundary_vector_space = VectorFunctionSpace(omega_b, 'Lagrange', 1)

    v, h = TestFunctions(state_space)
    state = Function(state_space)
    theta, phi = split(state)
    theta_bc = DirichletBC(state_space.sub(0), theta_b_3d, Boundary())

    def __init__(self, theta_n=theta_n_default_3d, phi_n=phi_n_default_3d, **kwargs):
        self.a = 0.006
        self.alpha = 0.333
        self.ka = 1
        self.b = 0.025
        self.beta = 1
        self.theta_n = theta_n
        self.phi_n = phi_n
        self.theta_b = theta_b_3d  # Warning! Might be ambiguous
        self._r = project(
            Expression('a * (theta_n + theta_b)',
                       degree=3,
                       a=self.a, theta_n=self.theta_n,
                       beta=self.beta, theta_b=self.theta_b
                       ),
            self.simple_space)
        for key, val in kwargs:
            setattr(self, key, val)


_n_2d = FacetNormal(DefaultValues2D.omega)
_n_3d = FacetNormal(DefaultValues3D.omega)


def partial_n_3d(x):
    return dot(_n_3d, grad(x))


def partial_n_2d(x):
    return dot(_n_2d, grad(x))
