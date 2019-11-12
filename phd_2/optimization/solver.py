# noinspection PyUnresolvedReferences
from dolfin import (
    Function, inner, grad, solve, interpolate, dx, ds,
    Constant, DirichletBC)
from ufl import dot

from phd_2.optimization.default_values import DefaultValues, Boundary, partial_n, _n


class SolveDirect(DefaultValues):

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
            - self.a * inner(dot(_n, grad(self.theta)), self.v) * ds \
            - self.alpha * inner(self.phi_n, self.h) * ds
        solve(boundary_problem == 0, self.state, self.theta_bc)
        return self.state

#  WIP
#
# class SolveOptimization(SolveDirect):
#     def __init__(self, *args, phi_n=Constant(0.1), **kwargs):
#         super().__init__(*args, **kwargs)
#         self.p1, self.p2 = TrialFunctions(super().state_space)
#         self.conjugate = Function(super().state_space)
#         self.quality_history = []
#         self.phi_n_derivative = phi_n
#         self.p_1, self.p_2 = Constant(0), Constant(0)
#
#     def quality(self):
#         quality = assemble(0.5 * (self.state[0] - self.theta_0) ** 2 * ds(self.omega)
#                            + self.epsilon * 0.5 * self.phi_n_derivative ** 2 * ds(self.omega))
#         self.quality_history.append(quality)
#         return quality
#
#     def solve_conjugate(self):
#         if not self.theta_0:
#             print('Set theta_0 first!')
#             return
#         conjugate_problem = \
#             self.a * inner(grad(self.p1), grad(self.v)) * dx + \
#             self.alpha * inner(grad(self.p2), grad(self.h)) * dx + \
#             4 * self.ka * self.state[0] ** 3 * inner(self.b * self.p1 - self.p2, self.v) * dx \
#             + self.beta * inner(self.p1, self.v) * ds + \
#             self.ka * inner(self.p2 - self.b * self.p1, self.h) * dx
#         j_theta = - (self.state[0] - self.theta_0) * self.v * ds
#         solve(conjugate_problem == j_theta, self.conjugate)
#         return self.conjugate.split()
