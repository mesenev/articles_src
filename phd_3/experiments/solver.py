# noinspection PyUnresolvedReferences
from dolfin import *
from dolfin import dx, ds

from default_values import DefaultValues3D


class Problem:

    def __init__(self, default_values=DefaultValues3D()):
        self.def_values = default_values

    def solve_boundary(self):
        v, h = self.def_values.v, self.def_values.h
        theta, psi = self.def_values.theta, self.def_values.psi
        a, b, ka, alpha, theta_n = self.def_values.a, self.def_values.b, \
                                   self.def_values.ka, self.def_values.alpha, \
                                   self.def_values.theta_n
        psi_n = self.def_values.phi_n

        theta_equation = \
            a * inner(grad(theta), grad(v)) * dx \
            + b * ka * inner(theta ** 4, v) * dx \
            + a * ka / alpha * inner(theta, v) * dx

        theta_src = - a * theta_n * v * ds

        psi_equation = inner(grad(psi), grad(h)) * dx + v * psi * ds

        psi_src = psi_n * h * ds

        state = self.def_values.state
        solve(
            theta_equation + psi_equation - theta_src - psi_src == 0, state,
            form_compiler_parameters={"optimize": True, 'quadrature_degree': 3},
            solver_parameters={"newton_solver": {"maximum_iterations": 500}}
        )
        return state
