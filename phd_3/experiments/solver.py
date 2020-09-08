from dolfin import *
# noinspection PyUnresolvedReferences
from dolfin import dx, ds

from default_values import DefaultValues3D


class Problem:

    def __init__(self, default_values=DefaultValues3D()):
        self.def_values = default_values
        self.theta, self.psi = None, None
        self.p1, self.p2 = None, None
        self.control = default_values.init_control
        self.ds = default_values.dss

    def solve_boundary(self):
        v = self.def_values.v
        ds = self.ds
        theta = self.def_values.theta
        a, b, ka, alpha, gamma, r = self.def_values.a, self.def_values.b, \
                                    self.def_values.ka, self.def_values.alpha, self.def_values.gamma, self.def_values.r
        theta_n, psi_n = self.def_values.theta_n, self.def_values.psi_n
        theta_b = self.def_values.theta_b
        psi = TrialFunction(self.def_values.simple_space)

        psi_equation = alpha * inner(grad(psi), grad(v)) * dx + gamma * v * psi * ds(1)
        psi_src = r * v * ds(1) + psi_n * v * ds(2)

        psi = Function(self.def_values.simple_space)
        solve(psi_equation == psi_src, psi)

        theta_equation = \
            a * inner(grad(theta), grad(v)) * dx \
            + b * ka * inner(theta ** 4, v) * dx \
            + a * ka / alpha * inner(theta, v) * dx \
            + inner(theta, v) * ds(1)

        theta_src = (a * theta_n + theta_b) * v * ds(1) + inner(psi, v) * dx

        solve(
            theta_equation - theta_src == 0, theta,
            form_compiler_parameters={"optimize": True, 'quadrature_degree': 3},
            solver_parameters={"newton_solver": {"maximum_iterations": 500}}
        )
        self.theta, self.psi = theta, psi
        return theta, psi

    def solve_conjugate(self):
        # TODO: test
        theta, psi = self.theta, self.psi
        v = self.def_values.v
        a, b, ka, alpha, = self.def_values.a, self.def_values.b, \
                           self.def_values.ka, self.def_values.alpha
        gamma = self.def_values.r
        p1 = TrialFunction(self.def_values.simple_space)

        p1_equation = a * inner(grad(p1), grad(v)) * dx \
                      + ka * inner((4 * b * theta + a / alpha) * p1, v) * dx
        p1_src = 0
        p1 = Function(self.def_values.simple_space)
        solve(p1_equation == p1_src, p1)

        p2 = TrialFunction(self.def_values.simple_space)
        p2_equation = alpha * inner(grad(p2), grad(v)) * dx \
                      + gamma * inner(p2, v) * ds(0)
        p2_src = 0

        p2 = Function(self.def_values.simple_space)
        solve(p2_equation == p2_src, p2)

        self.p1, self.p2 = p1, p2
        return p1, p2

    def update_control(self):
        # interpolation prevents nesting
        # TODO: test
        new_control = interpolate(
            Expression(
                'u - lm * (eps * u - p_2)',
                element=self.def_values.simple_space.ufl_element(),
                u=self.control, lm=self.def_values.lmbd,
                p_2=self.p2, eps=self.def_values.epsilon
            ), self.def_values.simple_space
        )
        self.control = new_control
        return new_control
