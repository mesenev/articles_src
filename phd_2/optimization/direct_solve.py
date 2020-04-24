# noinspection PyUnresolvedReferences
from dolfin import dx, ds
from dolfin import *

from default_values import DefaultValues3D
from solver import SolveBoundary


class DirectSolve(DefaultValues3D):
    finite_element = SolveBoundary.finite_element
    state_space = FunctionSpace(SolveBoundary.omega, MixedElement([finite_element]*4))
    z1, z2, x1, x2 = TestFunctions(state_space)
    state = Function(state_space)
    theta, phi, p1, p2 = split(state)

    def solve_direct(self):
        v, h = self.z1, self.z2
        theta_equation = \
            self.a * inner(grad(self.theta), grad(v)) * dx \
            + self.a * self.theta * v * ds + \
            + self.b * self.ka * inner(self.theta ** 4 - self.phi, v) * dx

        theta_src = self._r * v * ds

        phi_equation = \
            self.alpha * inner(grad(self.phi), grad(h)) * dx \
            + self.alpha * self.phi * h * ds \
            + self.ka * inner(self.phi - self.theta ** 4, h) * dx

        phi_src = self.p2 * h * ds

        v, h = self.x1, self.x2
        conjugate_theta = \
            self.a * inner(grad(self.p1), grad(v)) * dx \
            + self.a * self.p1 * v * ds + \
            + 4 * self.b * self.ka * inner(self.p1, self.theta ** 3 * v) * dx \
            - 4 * self.ka * inner(self.p2, self.theta ** 3 * v) * dx

        conjugate_phi = \
            self.alpha * inner(grad(self.p2), grad(h)) * dx \
            + self.alpha * self.p2 * h * ds \
            - self.b * self.ka * inner(self.p1, h) * dx \
            + self.ka * inner(self.p2, h) * dx

        j_theta = - (self.theta - self.theta_b) * v * ds

        solve(
            theta_equation + phi_equation - theta_src - phi_src + conjugate_theta + conjugate_phi - j_theta == 0,
            self.state,
        )
        return self.state
