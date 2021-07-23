from dolfin import dx, ds
from dolfin import *


class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary


mesh = UnitSquareMesh(32, 32)


class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] > 1.0 - DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS) and on_boundary


class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] < DOLFIN_EPS or x[1] < DOLFIN_EPS) and on_boundary


sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
sub_domains.set_all(0)
inflow = Inflow()
outflow = Outflow()

inflow.mark(sub_domains, 1)
outflow.mark(sub_domains, 2)
dss = ds(subdomain_data=sub_domains, domain=mesh)
V = FunctionSpace(mesh, "CG", 1)

g = Constant(1.0)
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("x[0]*sin(x[1])", degree=2)
F = inner(grad(u), grad(v)) * dx
f = g * v * dss(1) + f * v * dss(2)
u = Function(V)
solve(F == f, u)
print(min(u.vector()), max(u.vector()), sep='\n')
