from dolfin import *
from dolfin.cpp.mesh import *

def solve_boundary():
    # Create mesh and define function space
    omega = UnitCubeMesh(8, 8, 8)
    fem_space = FiniteElement("CG", omega.ufl_cell(), 1)
    Y = FunctionSpace(omega, fem_space * fem_space)

    theta, phi = TrialFunction(Y)
    v, h = TestFunctions(Y)

    # Define boundary condition
    w = Function(Y)
    a = inner(grad(theta), grad(v))*dx \
        + inner(grad(phi), grad(h))*dx
    L = v*ds

    solve(a == L, w)
    print('successfully solved.')
