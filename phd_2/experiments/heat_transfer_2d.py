from dolfin import *
from dolfin.cpp.common import set_log_level
from dolfin.cpp.function import near
from dolfin.cpp.mesh import UnitSquareMesh, BoundaryMesh
set_log_level(50)

def solve_boundary():
    omega = UnitSquareMesh(12, 12)
    # TODO: https://fenicsproject.org/qa/3074/get-function-values-on-boundaries/
    # Current 'boundaries' trick is not working :(
    Right = AutoSubDomain(lambda x, on_bnd: near(x[0], 1) and on_bnd)
    Top = AutoSubDomain(lambda x, on_bnd: near(x[1], 1) and on_bnd)
    a = 0.006
    alpha = 0.3333
    ka = 1
    b = 0.025
    epsilon = 0
    p = dolfin.Point(0.5, 0.5)
    lmbd = 0.5
    f = Expression('(x[0]+x[1])/2', degree=2)
    finite_element = FiniteElement("CG", omega.ufl_cell(), 2)
    state_space = FunctionSpace(omega, finite_element * finite_element)
    simple_fs = FunctionSpace(omega, finite_element)
    boundary_func = Function(simple_fs)
    DirichletBC(simple_fs, 1, Right).apply(boundary_func.vector())
    state = Function(state_space)
    adjoint = Function(state_space)
    theta, phi = split(state)
    p_1, p_2 = TrialFunctions(state_space)
    v, h = TestFunctions(state_space)
    theta_0 = None
    u_control = Expression("0.1*x[0]+0.1", degree=2)
    theta_b = interpolate(Expression("sin(x[0])*x[1]/2", degree=2), simple_fs)

    quality_history = []
    def quality():
        quality_history.append(assemble(0.5 * (theta_k-theta_0)**2 * ds(omega)))
        #+ epsilon * 0.5 * u_control**2*ds(omega)))
        return quality_history[-1]
    def set_and_solve_boundary():
        Boundary_problem = a * inner(grad(theta), grad(v)) * dx + alpha * inner(grad(phi), grad(h)) * dx \
        + b * inner((theta ** 4 - phi), v) * dx \
        + ka * inner((phi - theta ** 4), h) * dx \
        - f * inner(theta - theta_b, v) * ds - u_control * h * ds
        solve(Boundary_problem == 0, state)
        return state.split()
    theta_k, phi_k = set_and_solve_boundary()
    theta_0 = interpolate(theta_k, FunctionSpace(omega, finite_element))
    print(quality())

    def set_and_solve_adjoint():
        Adjoint_problem = a * inner(grad(p_1), grad(v))*dx + alpha * inner(grad(p_2), grad(h))*dx \
            + 4 * ka * theta_k**3 * inner(b*p_1 - p_2, v) * dx \
            + f * inner(p_1, v) * ds \
            + ka * inner(p_2 - b * p_1, h) * dx
        J_theta = theta_k* (theta_k - theta_0) * v * ds
        solve(Adjoint_problem == J_theta, adjoint)
        return adjoint.split()


    u_control = Expression('0.1*x[0]+0.3', degree=2)

    theta_k, phi_k = set_and_solve_boundary()
    print(quality())
    p_1_k, p_2_k = set_and_solve_adjoint()
    middle_point_history = []
    for _ in range(98):
        u_control = Expression('u - lmbd*(p_2+eps*u)', u=u_control, lmbd=lmbd, p_2=p_2_k, eps=epsilon, degree=2)
        theta_k, phi_k = set_and_solve_boundary()
        print('{} iteration, quality functional value: {}'.format(_, quality()))
        p_1_k, p_2_k = set_and_solve_adjoint()
        middle_point_history.append((theta_k(p), phi_k(p), p_1_k(p), p_2_k(p)))
    print('Quality diff: {}'.format(max(quality_history) - min(quality_history)))
    B = interpolate(u_control, simple_fs)
    print(B.vector()[simple_fs==1])
    u_control = Expression('0.1*x[0]+0.1', degree=2)
    theta_k,phi_k = set_and_solve_boundary()
    print(quality())
    return


solve_boundary()
    # Save solution in VTK format
    # file_local = File("neumann_poisson.pvd")
    # file_local << theta