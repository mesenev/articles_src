from dolfin import *
import matplotlib.pyplot as plt

from utilities import draw_simple_graphic

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True


omega = IntervalMesh(20, 0, 1)
omega_b = BoundaryMesh(omega, 'exterior')

finite_element = FiniteElement("CG", omega.ufl_cell(), 1)

state_space = FunctionSpace(omega, finite_element * finite_element)
simple_space = FunctionSpace(omega, finite_element)
vector_space = VectorFunctionSpace(omega, 'CG', 1)
boundary_vector_space = VectorFunctionSpace(omega_b, 'CG', 1)
boundary_simple_space = FunctionSpace(omega_b, 'CG', 1)

v = TestFunction(simple_space)
bc = DirichletBC(
    omega, Constant(0.0),
    lambda x: x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
)
state = Function(state_space)
theta, psi = Function(simple_space), Function(simple_space)

a = 0.6
alpha = 0.333
ka = 1
b = 0.025
beta = 1
gamma = 1
r = 1

psi_equation = alpha * inner(grad(psi), grad(v)) * dx + gamma * v * psi * ds(DIRICHLET)
psi_src = r * v * ds(DIRICHLET) + psi_n * v * ds(NEWMAN)

psi = Function(def_values.simple_space)
solve(psi_equation == psi_src, psi)

# Create mesh and define function space
mesh =
V = FunctionSpace(mesh, "CG", 2)


# Define Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class Source(Expression):
    def eval(self, values, x):
        values[0] = 4.0 * pi ** 4 * sin(pi * x[0]) * sin(pi * x[1])


# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define normal component, mesh size and right-hand side
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(123)
a = -inner(grad(u), grad(v)) * dx + inner(u, v) * dx
L = f * v * dx

# Penalty parameter
alpha = Constant(8.0)
u = Function(V)
solve(a == L, u, bc)

# Save solution to file
file = File("biharmonic.pvd")
file << u

# plotting solution
plot(u)
plt.savefig('123.png')

# Plot solution
