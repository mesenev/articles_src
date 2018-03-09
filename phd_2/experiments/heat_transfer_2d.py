from dolfin import *
import dolfin.cpp.mesh


def solve_boundary():
    mesh = dolfin.cpp.mesh.UnitSquareMesh.create(64, 64, dolfin.cpp.mesh.CellType.Type_quadrilateral)
    p1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, p1 * p1)
    (theta, phi) = TrialFunction(W)
    (v, h) = TestFunctions(W)
    f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
    g = Expression("-sin(5*x[0])", degree=2)
    a = inner(grad(theta), grad(v))*dx + inner(grad(phi), grad(h))*dx + theta * h * dx
    L = f * v * ds + g * v * ds

    # Compute solution
    w = Function(W)
    solve(a == L, w)
    return w

    # Save solution in VTK format
    # file_local = File("neumann_poisson.pvd")
    # file_local << theta