# noinspection PyUnresolvedReferences
from dolfin import (
    FunctionSpace, Function, split, TestFunctions, FiniteElement,
    Expression, Constant, DirichletBC, FacetNormal, project, inner,
    grad, interpolate, solve, dx, ds, assemble,
    TrialFunctions)

from phd_2.optimization.default_values import DefaultValues3D

_MAX_ITERATIONS = 10 ** 8


class SolveBoundary(DefaultValues3D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def print_min_max_state(self):
        for i in self.state:
            f = interpolate(i, self.simple_space).vector()
            print("Minimum:", f.min(), "\t maximum:", f.max())

    # TODO: WRITE TESTS FOR BOUNDARY PROBLEM
    def solve_boundary(self):
        boundary_problem = \
            self.a * inner(grad(self.theta), grad(self.v)) * dx \
            + self.b * self.ka * inner(self.theta ** 4 - self.phi, self.v) * dx \
            + self.a * (self.theta - self._r) * self.v * ds \
            + self.alpha * inner(grad(self.phi), grad(self.h)) * dx \
            + self.ka * inner(self.phi - self.theta ** 4, self.h) * dx \
            - self.alpha * inner(self.phi - self.phi_n, self.h) * ds
        solve(boundary_problem == 0, self.state)
        # theta_equation = \
        #     self.a * inner(grad(self.theta), grad(self.v)) * dx \
        #     + self.a * self.theta * self.v * ds(self.omega) + \
        #     + self.b * self.ka * inner(self.theta ** 4 - self.phi, self.v) * dx
        # theta_src = self._r * self.v * ds(self.omega)
        # phi_equation = \
        #     self.alpha * inner(grad(self.phi), grad(self.h)) * dx \
        #     + self.alpha * self.phi * self.h * ds(self.omega) \
        #     + self.ka * inner(self.phi - self.theta ** 4, self.h) * dx
        # phi_src = self.phi_n * self.h * ds(self.omega)
        #
        # solve(theta_equation + phi_equation - theta_src - phi_src == 0, self.state)
        # return self.state
        return self.state


class SolveOptimization(SolveBoundary):
    _lambda = 0.1 ** 2
    p1, p2 = TrialFunctions(DefaultValues3D.state_space)
    conjugate = Function(DefaultValues3D.state_space)
    tau, nu = TestFunctions(DefaultValues3D.state_space)
    epsilon = 0.1 ** 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quality_history = []

    def solve_conjugate(self):
        theta, phi = self.state.split()
        v, h = self.tau, self.nu
        conjugate_problem = \
            self.a * inner(grad(self.p1), grad(v)) * dx \
            + 4 * self.b * self.ka * inner(self.p1, theta ** 3 * v) * dx \
            + self.a * self.beta * inner(self.p1, v) * ds \
            - 4 * self.ka * inner(self.p2, theta ** 3 * v) * dx \
            + self.alpha * inner(grad(self.p2), grad(h)) * dx \
            + self.ka * inner(self.p2, h) * dx \
            - self.b * self.ka * inner(self.p1, h) * dx
        j_theta = - (theta - self.theta_b) * v * ds
        solve(conjugate_problem == j_theta, self.conjugate)
        return self.conjugate

    def recalculate_phi_n(self):
        """ Interpolation for preventing Expression nesting """
        new_phi_n = interpolate(
            Expression(
                'u - lm * (eps * u - p_2)', element=self.simple_space.ufl_element(),
                u=self.phi_n, lm=self._lambda,
                p_2=self.conjugate.split()[1], eps=self.epsilon
            ),
            self.simple_space
        )
        self.phi_n = new_phi_n
        return new_phi_n

    def quality(self, add_to_story=True):
        quality = assemble(
            0.5 * (self.state[0] - self.theta_b) ** 2 * ds(self.omega)
            + self.epsilon * 0.5 * self.phi_n ** 2 * ds(self.omega)
        )
        if add_to_story:
            self.quality_history.append(quality)
        return quality

    def _gradient_step(self):
        self.solve_boundary()
        self.quality()
        self.solve_conjugate()
        self.recalculate_phi_n()

    def find_optimal_control(self, iterations=_MAX_ITERATIONS, _lambda=None):
        if _lambda:
            self._lambda = _lambda

        for i in range(iterations):
            print(f'Iteration {i}', end='\t')
            self._gradient_step()
            print(f'quality: {self.quality_history[-1]}')
        return self.phi_n
