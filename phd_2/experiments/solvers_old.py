import dolfin
from dolfin import (
    FunctionSpace,
    Function, split, TestFunctions, inner, grad, solve, FiniteElement, dx, ds,
    TrialFunctions, Expression, assemble, interpolate, Constant
)
from dolfin.cpp.generation import UnitCubeMesh, UnitSquareMesh

# noinspection PyUnresolvedReferences

dolfin.cpp.log.set_log_level(50)


class SolveDirect:
    # omega = UnitCubeMesh(*[5] * 3)
    omega = UnitSquareMesh(10, 10)
    finite_element = FiniteElement("Lagrange", omega.ufl_cell(), 1)
    state_space = FunctionSpace(omega, finite_element * finite_element)
    simple_space = FunctionSpace(omega, finite_element)
    v, h = TestFunctions(state_space)

    def __init__(self, theta_0, q_b, a=0.006, alpha=0.333, ka=1, b=0.025):
        self.theta_0 = theta_0
        self.theta_b = q_b
        self.a = a
        self.alpha = alpha
        self.ka = ka
        self.b = b
        self._state_solve = Function(self.state_space)
        self.theta, self.phi = split(self._state_solve)
        self.state = None

    def print_min_max_functions(self):
        to_print = self.state
        for i in to_print:
            f = dolfin.interpolate(i, self.simple_space).vector()
            print("Minimum:", f.min(), "\t maximum:", f.max())


class SolveReverse(SolveDirect):
    lambda_ = 1000
    epsilon = 0.1 ** 20
    max_iterations = 10 ** 7

    def __init__(self, *args, phi_n=Constant(0.1), **kwargs):
        super().__init__(*args, **kwargs)
        self.p1, self.p2 = TrialFunctions(super().state_space)
        self.conjugate = Function(super().state_space)
        self.quality_history = []
        self.phi_n_derivative = phi_n
        self.p_1, self.p_2 = Constant(0), Constant(0)

    # TODO: WRITE TESTS FOR BOUNDARY
    def solve_boundary_with_phi_n_der(self):
        boundary_problem = \
            self.a * inner(grad(self.theta), grad(self.v)) * dx \
            + self.alpha * inner(grad(self.phi), grad(self.h)) * dx \
            + self.b * self.ka * inner(self.theta ** 4 - self.phi, self.v) * dx \
            + self.ka * inner(self.phi - self.theta ** 4, self.h) * dx \
            + self.beta * inner(self.theta - self.theta_0, self.v) * ds \
            - self.alpha * self.phi_n_derivative * self.h * ds
        solve(boundary_problem == 0, self._state_solve)
        self.state = self._state_solve.split()
        return self.state

    def recalculate_phi_n_derivative(self):
        self.phi_n_derivative = interpolate(
            Expression('u + lmbd*(p_2 - eps*u)', u=self.phi_n_derivative,
                       lmbd=self.lambda_, p_2=self.p_2, eps=self.epsilon, degree=3), self.simple_space
        )
        return

    def quality(self):
        quality = assemble(0.5 * (self.state[0] - self.theta_0) ** 2 * ds(self.omega)
                           + self.epsilon * 0.5 * self.phi_n_derivative ** 2 * ds(self.omega))
        self.quality_history.append(quality)
        return quality

    def solve_conjugate(self):
        if not self.theta_0:
            print('Set theta_0 first!')
            return
        conjugate_problem = \
            self.a * inner(grad(self.p1), grad(self.v)) * dx + \
            self.alpha * inner(grad(self.p2), grad(self.h)) * dx + \
            4 * self.ka * self.state[0] ** 3 * inner(self.b * self.p1 - self.p2, self.v) * dx \
            + self.beta * inner(self.p1, self.v) * ds + \
            self.ka * inner(self.p2 - self.b * self.p1, self.h) * dx
        j_theta = - (self.state[0] - self.theta_0) * self.v * ds
        solve(conjugate_problem == j_theta, self.conjugate)
        return self.conjugate.split()

    def solve_reverse(self, iterations=None, tolerance=None):
        if not iterations and not tolerance:
            iterations = 10
            tolerance = 1
        elif not iterations:
            iterations = self.max_iterations
        elif not tolerance:
            tolerance = 1
        iteration = 0
        while self.quality() > tolerance and iteration < iterations:
            iteration += 1
            print('Iteration:', str(iteration).ljust(len(str(iterations))), '\tQuality:', self.quality())
            self.solve_boundary_with_phi_n_der()
            self.p_1, self.p_2 = self.solve_conjugate()
            self.recalculate_phi_n_derivative()
