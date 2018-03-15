import dolfin
from dolfin.cpp.mesh import SubDomain, UnitSquareMesh
from dolfin import *

def solve_boundary():
    omega = UnitSquareMesh(12, 12)
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["optimize"] = True
    parameters["ghost_mode"] = "shared_facet"
    V = FunctionSpace(omega, "CG", 2)

    class DirichletBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class Source(UserExpression):
        def eval(self, values, x):
            values[0] = 4.0 * pi ** 4 * sin(pi * x[0]) * sin(pi * x[1])

    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, DirichletBoundary())
    u = TrialFunction(V)
    v = TestFunction(V)
    h = CellDiameter(omega)
    h_avg = (h('+') + h('-'))/2.0
    n = FacetNormal(omega)
    f = Source(degree=2)
    alpha = Constant(8.0)

    a = inner(div(grad(u)), div(grad(v))) * dx \
        - inner(avg(div(grad(u))), jump(grad(v), n)) * dS \
        - inner(jump(grad(u), n), avg(div(grad(v)))) * dS \
        + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS

    L = f*v*dx
    u = Function(V)
    solve(a == L, u, bc)
    return
