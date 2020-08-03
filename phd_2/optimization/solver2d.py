# noinspection PyUnresolvedReferences
from dolfin import (
    FunctionSpace, Function, split, TestFunctions, FiniteElement,
    Expression, Constant, DirichletBC, FacetNormal, project, inner,
    grad, interpolate, solve, dx, ds, assemble,
    TrialFunctions, VectorFunctionSpace)
from dolfin.cpp.generation import UnitSquareMesh
from dolfin.cpp.mesh import BoundaryMesh, SubDomain

_MAX_ITERATIONS = 10 ** 8


# Define Dirichlet boundary
class DirichletBoundary(SubDomain):
    # noinspection PyMethodOverriding
    def inside(self, x, on_boundary):
        return on_boundary


class DefaultValues2D:
    omega = UnitSquareMesh(100, 100)
    omega_b = BoundaryMesh(omega, 'exterior')
    finite_element = FiniteElement("Lagrange", omega.ufl_cell(), 1)

    state_space = FunctionSpace(omega, finite_element * finite_element)
    simple_space = FunctionSpace(omega, finite_element)
    vector_space = VectorFunctionSpace(omega, 'Lagrange', 2)
    boundary_simple_space = FunctionSpace(omega_b, 'Lagrange', 1)

    v, h = TestFunctions(state_space)
    state = Function(state_space)
    theta, phi = split(state)

    def __init__(self, theta_n, phi_n, theta_b, **kwargs):
        self.a = 0.92
        self.alpha = 0.0333
        self.ka = 1
        self.b = 0.19
        self.beta = 1
        self.theta_n = theta_n
        self.phi_n = phi_n
        self.theta_b = theta_b  # Warning! Might be ambiguous
        self._r = None
        self.theta_bc = DirichletBC(self.state_space.sub(0), theta_b, DirichletBoundary())
        self.recalculate_r()
        for key, val in kwargs.items():
            setattr(self, key, val)

    def recalculate_r(self):
        self._r = project(
            Expression(
                'a * (theta_n + theta_b)',
                degree=3,
                a=self.a, theta_n=self.theta_n,
                beta=self.beta, theta_b=self.theta_b
            ),
            self.simple_space)


class SolveBoundary(DefaultValues2D):

    def print_min_max_state(self):
        for i in self.state:
            f = interpolate(i, self.simple_space).vector()
            print("Minimum:", f.min(), "\t maximum:", f.max())

    def solve_boundary(self):
        theta_equation = \
            self.a * inner(grad(self.theta), grad(self.v)) * dx \
            + self.a * self.theta * self.v * ds + \
            + self.b * self.ka * inner(self.theta ** 4 - self.phi, self.v) * dx
        theta_src = self._r * self.v * ds
        phi_equation = \
            self.alpha * inner(grad(self.phi), grad(self.h)) * dx \
            + self.alpha * self.phi * self.h * ds \
            + self.ka * inner(self.phi - self.theta ** 4, self.h) * dx
        phi_src = self.phi_n * self.h * ds
        solve(
            theta_equation + phi_equation - theta_src - phi_src == 0, self.state,
            form_compiler_parameters={"optimize": True, 'quadrature_degree': 3}
        )
        return self.state

    def target_diff(self):
        return interpolate(
            Expression(
                'pow(theta - tb, 2)', degree=3,
                theta=self.state.split()[0], tb=self.theta_b
            ), self.simple_space
        )


class SolveOptimization(SolveBoundary):
    _lambda = 0.1 ** 2
    p1, p2 = TrialFunctions(DefaultValues2D.state_space)
    conjugate = Function(DefaultValues2D.state_space)
    tau, nu = TestFunctions(DefaultValues2D.state_space)
    epsilon = 0.1 ** 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quality_history = []

    def solve_conjugate(self):
        theta, phi = self.state.split()
        v, h = self.tau, self.nu
        conjugate_theta = \
            self.a * inner(grad(self.p1), grad(v)) * dx \
            + self.a * self.p1 * v * ds + \
            + 4 * self.b * self.ka * inner(self.p1, theta ** 3 * v) * dx \
            - 4 * self.ka * inner(self.p2, theta ** 3 * v) * dx
        conjugate_phi = \
            self.alpha * inner(grad(self.p2), grad(h)) * dx \
            + self.alpha * self.p2 * h * ds \
            - self.b * self.ka * inner(self.p1, h) * dx \
            + self.ka * inner(self.p2, h) * dx
        j_theta = - (theta - self.theta_b) * v * ds
        solve(
            conjugate_theta + conjugate_phi == j_theta, self.conjugate,
            form_compiler_parameters={"optimize": True, 'quadrature_degree': 3}
        )
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
            0.5 * (self.state.split()[0] - self.theta_b) ** 2 * dx
            # + self.epsilon * 0.5 * self.phi_n ** 2 * ds(self.omega)
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
            self._lambda += 0.5

            self._gradient_step()
            diff = self.quality_history[-2] - self.quality_history[-1] if len(self.quality_history) > 1 else 0
            print(f'Iteration {i},\tquality: {self.quality_history[-1]},\t{diff}')
            if diff < 0:
                print('warning')
                self._lambda -= 10
        return self.phi_n
