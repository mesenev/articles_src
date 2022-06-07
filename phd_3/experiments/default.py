# noinspection PyUnresolvedReferences
from dolfin import dx, ds
from dolfin import *
from mshr import *
from dolfin.cpp.common import DOLFIN_EPS
from dolfin.cpp.mesh import SubDomain
from mshr.cpp import Rectangle, generate_mesh

from phd_3.experiments.consts import NEWMAN, DIRICHLET
from utilities import print_3d_boundaries_on_cube, get_normal_derivative_3d, Wrapper, print_2d, NormalDerivativeZ

default_values = dict(
    a=0.333,
    alpha=0.6,
    ka=1,
    b=0.025,
    beta=1,
    gamma=1,
    epsilon=0.1 ** 10,
    theta_b=Expression('0.1 + x[2] / 2', degree=3),
    folder='default',
)


def der(x):
    return get_normal_derivative_3d(x, default_values['simple_space'], default_values['vector_space'])


def setup_spaces(d: dict, n=25):
    omega = UnitCubeMesh(*([n] * 3))
    finite_element = FiniteElement("CG", omega.ufl_cell(), 1)
    omega_b = BoundaryMesh(omega, 'exterior')
    d.update({
        'omega': omega,
        'finite_element': finite_element,
        'state_space': FunctionSpace(omega, finite_element * finite_element),
        'simple_space': FunctionSpace(omega, finite_element),
        'vector_space': VectorFunctionSpace(omega, 'CG', 1),
        'boundary_vector_space': VectorFunctionSpace(omega_b, 'CG', 1),
        'boundary_simple_space': FunctionSpace(omega_b, 'CG', 1),

    })


setup_spaces(default_values, 25)
# default_values['phi_n'] = Function(default_values['simple_space'], f'{default_values["folder"]}/phi_n.xml')
# default_values['phi_n'] = interpolate(default_values['phi_n'], default_values['boundary_simple_space'])


def solve_direct():
    v, h = TestFunctions(default_values['state_space'])
    state = Function(default_values['state_space'])
    theta, phi = split(state)
    a, alpha, b, ka, gamma, epsilon, theta_b, beta = (
        default_values['a'], default_values['alpha'],
        default_values['b'], default_values['ka'],
        default_values['gamma'], default_values['epsilon'],
        default_values['theta_b'], default_values['beta'],
    )

    theta_equation = \
        a * inner(grad(theta), grad(v)) * dx \
        + beta * theta * v * ds + \
        + b * ka * inner(theta ** 4 - phi, v) * dx
    theta_src = beta * inner(theta_b, v) * ds
    phi_equation = \
        alpha * inner(grad(phi), grad(h)) * dx \
        + gamma * phi * h * ds \
        + ka * inner(phi - theta ** 4, h) * dx
    phi_src = gamma * theta_b ** 4 * h * ds
    solve(
        theta_equation + phi_equation - theta_src - phi_src == 0, state,
        form_compiler_parameters={"optimize": True, 'quadrature_degree': 3},
        solver_parameters={"newton_solver": {"linear_solver": "mumps"}}
    )
    File(f'{default_values["folder"]}/state.xml') << state


solve_direct()

state = Function(default_values['state_space'], f'{default_values["folder"]}/state.xml')
theta, phi = state.split()
theta, phi = project(theta, default_values['simple_space']), project(phi, default_values['simple_space'])
print_3d_boundaries_on_cube(theta, name='theta', folder=default_values['folder'])
print_3d_boundaries_on_cube(phi, name='phi', folder=default_values['folder'])
phi_n = der(phi)
# File(f'{default_values["folder"]}/phi_n.xml') << phi_n
theta_n = der(theta)
print_3d_boundaries_on_cube(phi_n, name='phi_n', folder=default_values['folder'])
print_3d_boundaries_on_cube(theta_n, name='theta_n', folder=default_values['folder'])
gamma = project(
    (default_values['alpha'] * phi_n / (phi - default_values["theta_b"] ** 4)),
    default_values['simple_space'],
)
print_3d_boundaries_on_cube(gamma, name='gamma', folder=default_values['folder'])
