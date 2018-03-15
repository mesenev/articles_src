from dolfin import *
from dolfin.cpp.mesh import UnitSquareMesh


def sample():
    mesh = UnitSquareMesh(32, 32)

    BDM = FiniteElement("BDM", mesh.ufl_cell(), 1)
    DG = FiniteElement("DG", mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, BDM * DG)
    (sigma, u) = TrialFunctions(W)
    (tau, v) = TestFunctions(W)

    # Define source functions
    f = Expression("10",degree=2)
    g = Expression("sin(5.0*x[0])", degree=2)

    # Define variational form
    a = (dot(sigma, tau) + dot(grad(u), tau) + dot(sigma, grad(v)))*dx
    L = - f*v*dx - g*v*ds

    # Define Dirichlet BC
    def boundary(x):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
    bc = DirichletBC(W.sub(1), 0.0, boundary)

    # Compute solution
    w = Function(W)
    solve(a == L, w, bc)
    return w.split()

