# noinspection PyUnresolvedReferences
from dolfin import (
    FunctionSpace, Function, split, TestFunctions, FiniteElement,
    Expression, Constant, DirichletBC, FacetNormal, project, inner,
    grad, interpolate, solve, dx, ds, assemble,
    TrialFunctions)

from phd_2.experiments.utilities import print_2d_boundaries, print_2d
from phd_2.optimization.default_values import DefaultValues

_MAX_ITERATIONS = 10 ** 8


class SolveBoundary(DefaultValues):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def print_min_max_functions(self):
        for i in self.state:
            f = interpolate(i, self.simple_space).vector()
            print("Minimum:", f.min(), "\t maximum:", f.max())

    # TODO: WRITE TESTS FOR BOUNDARY PROBLEM
    def solve_boundary(self):
        boundary_problem = \
            self.a * inner(grad(self.theta), grad(self.v)) * dx \
            + self.alpha * inner(grad(self.phi), grad(self.h)) * dx \
            + self.b * self.ka * inner(self.theta ** 4 - self.phi, self.v) * dx \
            + self.ka * inner(self.phi - self.theta ** 4, self.h) * dx \
            + self.a * (self.beta * self.theta - self._r) * self.v * ds \
            - self.alpha * inner(self.phi_n, self.h) * ds
        solve(boundary_problem == 0, self.state)
        return self.state


class SolveOptimization(SolveBoundary):
    _lambda = 0.1 ** 3
    p1, p2 = TrialFunctions(DefaultValues.state_space)
    conjugate = Function(DefaultValues.state_space)
    epsilon = 0.1 ** 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quality_history = []

    def solve_conjugate(self):
        theta, phi = self.state.split()
        conjugate_problem = \
            self.a * inner(grad(self.p1), grad(self.v)) * dx \
            + self.alpha * inner(grad(self.p2), grad(self.h)) * dx \
            + 4 * self.b * self.ka * inner(theta ** 3 * self.v, self.p1) * dx \
            - self.b * self.ka * inner(self.h, self.p2) * dx \
            - 4 * self.ka * inner(theta ** 3 * self.v, self.p2) * dx \
            + self.ka * inner(self.h, self.p2) * dx \
            + self.a * self.beta * inner(self.p1, self.v) * ds
        j_theta = - (theta - self.theta_b) * self.v * ds
        solve(conjugate_problem == j_theta, self.conjugate)
        return self.conjugate

    def recalculate_phi_n(self):
        p1, p2 = self.conjugate.split()
        print_2d(p2, name='p2')
        print_2d(p1, name='p1')
        new_phi_n = interpolate(
            Expression('u - eps*lm*p_2', degree=2, u=self.phi_n, lm=self._lambda, p_2=p2, eps=1),
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
        phi_n_current = self.phi_n
        self.recalculate_phi_n()
        phi_n_diff = Expression('o - n', o=phi_n_current, n=self.phi_n, degree=2)
        print_2d_boundaries(phi_n_diff)

    def find_optimal_control(self, iterations=_MAX_ITERATIONS):
        for i in range(iterations):
            print(f'Iteration {i}')
            self._gradient_step()
        return self.phi_n
