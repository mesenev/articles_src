import itertools

from utilities import Normal

# noinspection PyUnresolvedReferences
from dolfin import dx, ds
from dolfin import *


class Normal(UserExpression):
    def __init__(self, mesh, dimension, **kwargs):
        self.mesh = mesh
        self.dimension = dimension
        super().__init__(**kwargs)

    def eval_cell(self, values, x, ufc_cell):
        values[0] = 0
        values[1] = 0
        for i in range(self.dimension):

            if x[i] < DOLFIN_EPS:
                values[i] = -1
            if x[i] + DOLFIN_EPS > 1:
                values[i] = 1
        for i, j in itertools.combinations(list(range(self.dimension)), 2):
            if abs(values[i]) == abs(values[j]) == 1:
                values[0] *= 0.5
                values[1] *= 0.5

    def value_shape(self):
        return (2,) if self.dimension == 2 else (3,)


omega = UnitSquareMesh(10, 10)
omega_b = BoundaryMesh(omega, 'exterior')
finite_element = FiniteElement("Lagrange", omega.ufl_cell(), 2)

state_space = FunctionSpace(omega, finite_element * finite_element)
simple_space = FunctionSpace(omega, finite_element)
vector_space = VectorFunctionSpace(omega, 'Lagrange', 2)
boundary_vector_space = VectorFunctionSpace(omega_b, 'Lagrange', 1)
boundary_simple_space = FunctionSpace(omega_b, 'Lagrange', 1)
normal = interpolate(Normal(omega, dimension=2), boundary_vector_space)
normal_2 = FacetNormal(omega_b)
theta = project(
    Expression('pow((x[0]-0.5), 2) - 0.5*x[1] + 0.75', element=simple_space.ufl_element()),
    simple_space
)
grad_t_b = interpolate(project(grad(theta), vector_space), boundary_vector_space)

theta_n = project(inner(grad_t_b, normal), boundary_simple_space)
v, h = TestFunctions(state_space)
state = Function(state_space)
a, b = split(state)

a_eq = inner(grad(a), grad(v)) * dx + inner(a ** 4 - b, v) * dx
b_eq = inner(grad(b), grad(h)) * dx + inner(b - a ** 4, h) * dx
a_src = 0.3 * v * ds
b_src =  h * ds
solve(a_eq + b_eq - a_src - b_src == 0, state)
