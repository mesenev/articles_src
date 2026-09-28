# docker run --rm -v "$(pwd)":/source -w /source/playground --entrypoint python3 ghcr.io/fenics/dolfinx/lab:stable diffuse.py
#
# Проверка решателя методом искусственных решений (MMS) для модели (1.1)-(1.3):
#   -d Δρ - μ ∇φ·∇ρ + μ/(εε0) |ρ|ρ = f,   -Δφ = ρ/(εε0),   ρ = φ = 0 на Γ   (E = -∇φ)
# Точное решение обязано удовлетворять тем же граничным условиям, т.е. φ_ex = 0 и Δφ_ex = 0 на Γ
# (ρ_ex = -εε0 Δφ_ex). Два таких решения:
#   A: φ_ex = g(x)g(y), g(t) = t - 2t³ + t⁴  (g = g'' = 0 при t = 0, 1)
#   B: φ_ex = sin(πx)sin(πy)/(2π²),  ρ_ex = εε0 sin(πx)sin(πy)
# Критерии качества: сходимость Ньютона, порядки сходимости ошибок при измельчении сетки
# (P1: ≈2 в L2, ≈1 в H1), неотрицательность ρ_h.
import sys

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

import numpy as np

import ufl
from ufl import dx, inner, grad, div
from basix.ufl import element, mixed_element
from dolfinx import default_real_type
from dolfinx.fem import Function, functionspace, dirichletbc, locate_dofs_topological, form, assemble_scalar
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_unit_square, exterior_facet_indices

# ================= ПАРАМЕТРЫ =================
# Для численного теста удобно масштабировать параметры к O(1),
# чтобы избежать плохой обусловленности матрицы Якоби.
d_val = 1.0
mu_val = 0.5
eps0_val = 1.0
eps_val = 1.0
eps_eps0 = eps_val * eps0_val

MESH_SIZES = [16, 32, 64, 128]
EXPECTED_RATE = {"L2": 2.0, "H1": 1.0}
RATE_TOL = 0.15

PETSC_OPTIONS = {
    "snes_type": "newtonls",
    "snes_rtol": 1e-10,
    "snes_atol": 1e-12,
    "snes_max_it": 50,
    "snes_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def phi_exact_poly(x, y):
    def g(t):
        return t - 2 * t ** 3 + t ** 4
    return g(x) * g(y)


def phi_exact_sin(x, y):
    return ufl.sin(ufl.pi * x) * ufl.sin(ufl.pi * y) / (2 * ufl.pi ** 2)


EXACT_SOLUTIONS = {"A": phi_exact_poly, "B": phi_exact_sin}


def solve(n, phi_exact=phi_exact_sin):
    omega = create_unit_square(MPI.COMM_WORLD, n, n, CellType.triangle)
    P1 = element("Lagrange", omega.basix_cell(), 1, dtype=default_real_type)
    V = functionspace(omega, mixed_element([P1, P1]))
    u = Function(V)
    rho, phi = ufl.split(u)
    w_rho, w_phi = ufl.TestFunctions(V)

    x, y = ufl.SpatialCoordinate(omega)
    phi_ex = phi_exact(x, y)
    rho_ex = -eps_eps0 * div(grad(phi_ex))

    # Правая часть f, вычисленная через UFL (автоматическое символьное дифференцирование)
    f_ex = - d_val * div(grad(rho_ex)) \
           - mu_val * inner(grad(phi_ex), grad(rho_ex)) \
           + (mu_val / eps_eps0) * abs(rho_ex) * rho_ex

    F_rho = d_val * inner(grad(rho), grad(w_rho)) * dx \
            - mu_val * inner(grad(phi), grad(rho)) * w_rho * dx \
            + (mu_val / eps_eps0) * abs(rho) * rho * w_rho * dx \
            - f_ex * w_rho * dx

    F_phi = inner(grad(phi), grad(w_phi)) * dx \
            - (1.0 / eps_eps0) * rho * w_phi * dx

    F = F_rho + F_phi  # Монолитная нелинейная система

    tdim = omega.topology.dim
    omega.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = exterior_facet_indices(omega.topology)
    bcs = [
        dirichletbc(ScalarType(0), locate_dofs_topological(V.sub(i), tdim - 1, boundary_facets), V.sub(i))
        for i in range(2)
    ]

    problem = NonlinearProblem(
        F, u, bcs=bcs, petsc_options_prefix=f"mms{n}_", petsc_options=PETSC_OPTIONS
    )
    problem.solve()
    snes = problem.solver

    # Интегралы с повышенной квадратурой, чтобы ошибка интегрирования не маскировала ошибку решения.
    # assemble_scalar считает только локальную часть сетки, поэтому суммируем по процессам.
    dxq = ufl.Measure("dx", domain=omega, metadata={"quadrature_degree": 8})

    def integral(expr):
        return omega.comm.allreduce(assemble_scalar(form(expr * dxq)), op=MPI.SUM)

    result = {
        "n": n,
        "h": 1.0 / n,
        "newton_its": snes.getIterationNumber(),
        "newton_reason": snes.getConvergedReason(),
        "residual": snes.getFunctionNorm(),
        "rho_min": omega.comm.allreduce(u.sub(0).collapse().x.array.min(), op=MPI.MIN),
        "fields": {"u": u, "rho_ex": rho_ex, "phi_ex": phi_ex},
    }
    for name, uh, uex in (("rho", rho, rho_ex), ("phi", phi, phi_ex)):
        e = uh - uex
        result[f"{name}_L2"] = np.sqrt(integral(e ** 2))
        result[f"{name}_H1"] = np.sqrt(integral(inner(grad(e), grad(e))))
        result[f"{name}_relL2"] = result[f"{name}_L2"] / np.sqrt(integral(uex ** 2))
    return result


def rates(results, key):
    return [
        np.log(a[key] / b[key]) / np.log(a["h"] / b["h"])
        for a, b in zip(results, results[1:])
    ]


def check(name, phi_exact):
    results = [solve(n, phi_exact) for n in MESH_SIZES]
    if MPI.COMM_WORLD.rank != 0:
        return True

    print(f"\n===== Тест {name} =====")
    print(f"{'N':>5} {'Newton':>6} {'|F|':>9} {'min rho_h':>10} "
          f"{'rho L2':>9} {'rho H1':>9} {'rho relL2':>9} "
          f"{'phi L2':>9} {'phi H1':>9} {'phi relL2':>9}")
    for r in results:
        print(f"{r['n']:>5} {r['newton_its']:>6} {r['residual']:>9.2e} {r['rho_min']:>10.2e} "
              f"{r['rho_L2']:>9.2e} {r['rho_H1']:>9.2e} {r['rho_relL2']:>9.2e} "
              f"{r['phi_L2']:>9.2e} {r['phi_H1']:>9.2e} {r['phi_relL2']:>9.2e}")

    print("\nНаблюдаемые порядки сходимости (между соседними сетками):")
    passed = True
    for name in ("rho", "phi"):
        for norm, expected in EXPECTED_RATE.items():
            rs = rates(results, f"{name}_{norm}")
            ok = abs(rs[-1] - expected) <= RATE_TOL
            passed &= ok
            print(f"  {name} {norm}: {' '.join(f'{r:5.2f}' for r in rs)}"
                  f"   (ожидается {expected:.1f}) {'PASS' if ok else 'FAIL'}")

    # Точное ρ неотрицательно (лемма 3.1); у P1-решения допустим отрицательный выброс порядка ошибки.
    rho_min = min(r["rho_min"] for r in results)
    ok = rho_min >= -results[-1]["rho_L2"]
    passed &= ok
    print(f"  min rho_h = {rho_min:.2e}   (ожидается >= 0) {'PASS' if ok else 'FAIL'}")
    return passed


def main():
    passed = all([check(name, phi_exact) for name, phi_exact in EXACT_SOLUTIONS.items()])
    if MPI.COMM_WORLD.rank == 0:
        print(f"\nИТОГ: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
