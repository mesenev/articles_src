# docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v "$(pwd)":/source -w /source/playground \
#     --entrypoint python3 ghcr.io/fenics/dolfinx/lab:stable linbo3.py [h_min_um] [h_max_um]
#
# Вычислительный эксперимент из разд. 4 статьи Бризицкий, Максимова, Веселова (ЖВМиМФ, 2026, т. 66, № 4):
# зарядка LiNbO3 электронным зондом, задача (4.1) с β(x), ρ|Γ = ψ ≠ 0, φ|Γ = 0.
#
#   -D Δρ - μ ∇φ·∇ρ + (μ/εε0) β|ρ|ρ = f,   -Δφ = ρ/εε0   в  (0,L)³,
#   ρ = ψ, φ = 0 на Γ.
#
# Решатель тот же, что в diffuse.py (смешанные P1×P1, Ньютон + MUMPS), плюс:
#  * обезразмеривание: x = L ξ, ρ = R r, φ = Φ p, Φ = R L²/(εε0);
#  * четверть куба: задача симметрична относительно x = L/2 и y = L/2,
#    на плоскостях симметрии — естественное условие ∂n = 0;
#  * сетка gmsh, сгущённая к источнику (L/2, L/2, 0);
#  * SUPG для уравнения ρ: дрейф доминирует, сеточное число Пекле ~10³;
#  * продолжение по коэффициенту диффузии: D·K, K = 10⁴ → 1.
import sys
from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

import numpy as np
import gmsh

import ufl
from ufl import inner, grad
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, geometry
from dolfinx.fem import (Function, Constant, Expression, functionspace, dirichletbc, locate_dofs_topological,
                         form, assemble_scalar)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io.gmsh import model_to_mesh
from dolfinx.mesh import locate_entities_boundary

# ---------- Таблица 1 ----------
L = 1e-4            # м
D = 1.915e-6        # м²/с
mu = 74e-6          # м²/(В·с)
eps = 30.0
eps0 = 8.85e-12     # Ф/м
d1 = 0.76e-6        # м
d2 = 0.75e-6        # м
eta = 0.244
f0 = 1.045e16       # Кл/(м³·с)
eps_eps0 = eps * eps0

# ---------- масштабы ----------
R = 1e3                         # Кл/м³, порядок ψ
PHI = R * L ** 2 / eps_eps0     # В
PE = mu * PHI / D               # μ R L² / (εε0 D) ≈ 1.46·10⁶
S = eps_eps0 / (mu * R ** 2)    # f → f εε0/(μ R²)

H_MIN = float(sys.argv[1]) * 1e-6 / L if len(sys.argv) > 1 else 0.25e-6 / L
H_MAX = float(sys.argv[2]) * 1e-6 / L if len(sys.argv) > 2 else 6e-6 / L
C_DC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0    # 0 — только SUPG
DC_OUTER = 4
K_STEPS = [1e4, 1e3, 3e2, 1e2, 3e1, 1e1, 3.0, 1.0]

OUT = Path("linbo3_out")

PETSC_OPTIONS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_rtol": 1e-9,
    "snes_atol": 1e-10,
    "snes_max_it": 60,
    "snes_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def build_mesh():
    """Четверть куба [0,½]×[0,½]×[0,1] (в единицах L), источник в начале координат."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("quarter")
    box = gmsh.model.occ.addBox(0, 0, 0, 0.5, 0.5, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [box], 1)

    src = gmsh.model.occ.addPoint(0, 0, 0)
    gmsh.model.occ.synchronize()
    fd = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fd, "PointsList", [src])
    ft = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(ft, "InField", fd)
    gmsh.model.mesh.field.setNumber(ft, "SizeMin", H_MIN)
    gmsh.model.mesh.field.setNumber(ft, "SizeMax", H_MAX)
    gmsh.model.mesh.field.setNumber(ft, "DistMin", (d1 + 3 * d2) / L)
    gmsh.model.mesh.field.setNumber(ft, "DistMax", 0.35)
    gmsh.model.mesh.field.setAsBackgroundMesh(ft)
    for opt in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber(f"Mesh.{opt}", 0)
    gmsh.model.mesh.generate(3)
    data = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    return data.mesh


def solve():
    omega = build_mesh()
    tdim = omega.topology.dim
    P1 = element("Lagrange", omega.basix_cell(), 1, dtype=default_real_type)
    V = functionspace(omega, mixed_element([P1, P1]))
    u = Function(V)
    r, p = ufl.split(u)
    w, q = ufl.TestFunctions(V)

    X, Y, Z = ufl.SpatialCoordinate(omega)          # X = (x - L/2)/L, Y = (y - L/2)/L, Z = z/L
    dist = ufl.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    beta = 0.5 + 2 * ufl.exp(-dist ** 2)
    src = S * f0 * (1 - eta) * ufl.exp(-(dist * L - d1) ** 2 / (2 * d2 ** 2))

    dx = ufl.Measure("dx", domain=omega, metadata={"quadrature_degree": 4})
    K = Constant(omega, ScalarType(K_STEPS[0]))
    eps_d = K / PE

    # Галёркин
    F_r = eps_d * inner(grad(r), grad(w)) * dx \
        - inner(grad(p), grad(r)) * w * dx \
        + beta * abs(r) * r * w * dx \
        - src * w * dx
    F_p = inner(grad(p), grad(q)) * dx - r * q * dx

    # SUPG: скорость b = -∇p (направление E), P1 ⇒ Δr = 0 на элементе
    h = ufl.CellDiameter(omega)
    b = -grad(p)
    bnorm = ufl.sqrt(inner(b, b) + 1e-20)
    sigma = 2 * beta * abs(r)
    tau = 1 / ufl.sqrt((2 * bnorm / h) ** 2 + 9 * (4 * eps_d / h ** 2) ** 2 + sigma ** 2)
    res = inner(b, grad(r)) + beta * abs(r) * r - src
    F_supg = tau * inner(b, grad(w)) * res * dx

    # изотропная диффузия, захватывающая разрывы (Codina): ε_dc = C h |res| / (2|∇r|)
    # коэффициент запаздывает: P0-функция, пересчитывается между решениями Ньютона
    gnorm = ufl.sqrt(inner(grad(r), grad(r)) + 1e-12)
    eps_dc = Function(functionspace(omega, ("DG", 0)))
    eps_dc_expr = Expression(h * ufl.sqrt(res ** 2 + 1e-12) / (2 * gnorm),
                             eps_dc.function_space.element.interpolation_points)
    F_dc = eps_dc * inner(grad(r), grad(w)) * dx

    F = F_r + F_p + F_supg + (F_dc if C_DC > 0 else 0)

    # Дирихле на гранях x=L, y=L (X=½, Y=½), z=0, z=L; на X=0, Y=0 — симметрия
    def dirichlet(x):
        return (np.isclose(x[0], 0.5) | np.isclose(x[1], 0.5)
                | np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0))

    facets = locate_entities_boundary(omega, tdim - 1, dirichlet)
    V_r, _ = V.sub(0).collapse()
    psi = Function(V_r)
    psi.interpolate(lambda x: 1.0 + np.where(np.isclose(x[2], 0.0),
                                             (np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])) ** 2, 0.0))
    bcs = [
        dirichletbc(psi, locate_dofs_topological((V.sub(0), V_r), tdim - 1, facets), V.sub(0)),
        dirichletbc(ScalarType(0), locate_dofs_topological(V.sub(1), tdim - 1, facets), V.sub(1)),
    ]

    problem = NonlinearProblem(F, u, bcs=bcs, petsc_options_prefix="linbo3_", petsc_options=PETSC_OPTIONS)
    n_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    if omega.comm.rank == 0:
        print(f"сетка: {omega.topology.index_map(tdim).size_global} тетраэдров, {n_dofs} неизвестных; "
              f"h_min = {H_MIN * L * 1e6:.2f} мкм, h_max = {H_MAX * L * 1e6:.1f} мкм; Pe = {PE:.3e}; C_dc = {C_DC}", flush=True)
    for k in K_STEPS:
        K.value = k
        problem.solve()
        for _ in range(DC_OUTER if C_DC > 0 else 0):
            eps_dc.interpolate(eps_dc_expr)
            eps_dc.x.array[:] *= C_DC
            problem.solve()
        rr = u.sub(0).collapse().x.array
        if omega.comm.rank == 0:
            print(f"  K = {k:8.1f}: Ньютон {problem.solver.getIterationNumber():2d} ит., "
                  f"ρ ∈ [{rr.min() * R:.4e}, {rr.max() * R:.4e}] Кл/м³", flush=True)
    return omega, u, beta, src


def evaluate(omega, f, pts):
    """Значения f в точках pts (N×3, безразмерные), NaN вне сетки."""
    tree = geometry.bb_tree(omega, omega.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    cells = geometry.compute_colliding_cells(omega, cand, pts)
    vals = np.full(len(pts), np.nan)
    idx, cell = [], []
    for i in range(len(pts)):
        c = cells.links(i)
        if len(c):
            idx.append(i)
            cell.append(c[0])
    vals[idx] = f.eval(pts[idx], np.array(cell, dtype=np.int32)).reshape(len(idx), -1)[:, 0]
    return vals


def report(omega, u, beta, src):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    rho = u.sub(0).collapse()
    phi = u.sub(1).collapse()
    rho_d = rho.x.array * R
    phi_d = phi.x.array * PHI

    dxq = ufl.Measure("dx", domain=omega, metadata={"quadrature_degree": 6})
    vol = omega.comm.allreduce(assemble_scalar(form(1.0 * dxq)), op=MPI.SUM)
    print(f"объём четверти = {vol:.4f} (ожидается 0.25)")

    f_max = f0 * (1 - eta)
    beta_min = 0.5 + 2 * np.exp(-1.5)
    M = max(2e3, np.sqrt(f_max * eps_eps0 / (mu * beta_min)))
    print(f"\nρ:  min = {rho_d.min():.4e},  max = {rho_d.max():.4e} Кл/м³")
    print(f"    оценка (3.1): m = 0, M = {M:.4e} Кл/м³ (в статье 1.732e5; ρ_max(COMSOL) = 1.4774e5)")
    print(f"φ:  min = {phi_d.min():.4e},  max = {phi_d.max():.4e} В (в статье от -800 до 0 В)")

    # профиль по глубине из (L/2, L/2, 0)
    z = np.linspace(0, 1, 801)
    line = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    prof_r = evaluate(omega, rho, line) * R
    prof_p = evaluate(omega, phi, line) * PHI
    np.savetxt(OUT / "profile.txt", np.column_stack([z * L, prof_r, prof_p]),
               header="z [m]   rho [C/m^3]   phi [V]")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
    ax[0].plot(z * L * 1e6, prof_r)
    ax[0].set(xlabel="z, мкм", ylabel="ρ, Кл/м³", title="ρ(L/2, L/2, z)", yscale="log")
    ax[1].plot(z * L * 1e6, prof_p)
    ax[1].set(xlabel="z, мкм", ylabel="φ, В", title="φ(L/2, L/2, z)")
    fig.savefig(OUT / "profile.png", dpi=150)

    # срезы на глубинах 5, 10, 15, 30 мкм, четверть отражается на весь квадрат
    n = 201
    s = np.linspace(0, 0.5, n)
    XX, YY = np.meshgrid(s, s, indexing="ij")
    for name, f, scale, unit in (("rho", rho, R, "Кл/м³"), ("phi", phi, PHI, "В")):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), layout="constrained")
        for ax, depth in zip(axes, (5, 10, 15, 30)):
            pts = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, depth * 1e-6 / L)])
            v = evaluate(omega, f, pts).reshape(n, n) * scale
            full = np.block([[v[::-1, ::-1], v[::-1, 1:]], [v[1:, ::-1], v[1:, 1:]]])
            g = (np.concatenate([-s[::-1], s[1:]]) + 0.5) * L * 1e6
            cs = ax.contourf(g, g, full.T, 40, cmap="jet")
            fig.colorbar(cs, ax=ax)
            ax.set(title=f"{name}, глубина {depth} мкм, {unit}", xlabel="x, мкм", ylabel="y, мкм", aspect="equal")
        fig.savefig(OUT / f"{name}_slices.png", dpi=130)

    print("\nглубина  ρ_max(срез)    φ(центр)     ")
    for depth in (5, 10, 15, 30):
        pts = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, depth * 1e-6 / L)])
        print(f"{depth:4d} мкм  {np.nanmax(evaluate(omega, rho, pts)) * R:.4e}  "
              f"{evaluate(omega, phi, np.array([[0, 0, depth * 1e-6 / L]]))[0] * PHI:.4e}")


def main():
    omega, u, beta, src = solve()
    if MPI.COMM_WORLD.rank == 0:
        report(omega, u, beta, src)


if __name__ == "__main__":
    main()
