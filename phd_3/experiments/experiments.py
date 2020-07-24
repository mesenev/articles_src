from dolfin import *

mesh = UnitSquareMesh(32, 32)
File("mesh.pvd") << mesh

V = FunctionSpace(mesh, "CG", 1)


class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary


g = Constant(1.0)
g = Constant(1.0)
u = Function(V)
v = TestFunction(V)
f = Expression("x[0]*sin(x[1])", degree=2)
F = inner(grad(u), grad(v)) * dx \
    + inner(u ** 4, v) * dx \
    + inner(u, v) * dx - f * v * ds

# Compute solution
solve(F == 0, u,
      solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})
print(min(u.vector()))
