# docker run --init -ti -p 8888:8888 dolfinx/lab:stable
import os
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType 

import numpy as np

from ufl import (ds, dx, inner, grad, div)
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot
from dolfinx.fem import Function, functionspace, dirichletbc, Constant, locate_dofs_topological, form, assemble_scalar
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square, exterior_facet_indices
import pyvista as pv
import matplotlib.pyplot as plt

# ================= ПАРАМЕТРЫ =================
# Для численного теста удобно масштабировать параметры к O(1), 
# чтобы избежать плохой обусловленности матрицы Якоби.
d_val  = 1.0
mu_val = 0.5
eps0_val = 1.0
eps_val  = 1.0
omega = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle)
P1 = element("Lagrange", omega.basix_cell(), 1, dtype=default_real_type)
V = functionspace(omega, mixed_element([P1, P1]))
u = Function(V)
rho, phi = ufl.split(u)
w_rho, w_phi = ufl.TestFunctions(V)

tdim = omega.topology.dim
omega.topology.create_connectivity(tdim - 1, tdim)

x, y = ufl.SpatialCoordinate(omega)
phi_ex  = x * (1 - x) * y * (1 - y)
rho_ex  = - eps_val * eps0_val * div(grad(phi_ex))  # Точно совпадает с формулой выше

# Правая часть f, вычисленная через UFL (автоматическое символьное дифференцирование)
f_ex = - d_val * div(grad(rho_ex)) \
       - mu_val * inner(grad(phi_ex), grad(rho_ex)) \
       + (mu_val / (eps_val * eps0_val)) * rho_ex ** 2
# print(f_ex)

F_rho = d_val * inner(grad(rho), grad(w_rho)) * dx \
        - mu_val * inner(grad(phi), grad(rho)) * w_rho * dx \
        + (mu_val / (eps_val * eps0_val)) * rho ** 2 * w_rho * dx \
        - f_ex * w_rho * dx

F_phi = inner(grad(phi), grad(w_phi)) * ufl.dx \
        - (1.0 / (eps_val * eps0_val)) * rho * w_phi * dx

F = F_rho + F_phi  # Монолитная нелинейная система


boundary_facets = exterior_facet_indices(omega.topology)
dofs_rho = locate_dofs_topological(V.sub(0), tdim-1, boundary_facets)
dofs_phi = locate_dofs_topological(V.sub(1), tdim-1, boundary_facets)

bc_rho = dirichletbc(value=ScalarType(0), dofs=dofs_rho, V=V.sub(0))
bc_phi = dirichletbc(value=ScalarType(0), dofs=dofs_phi, V=V.sub(1))
bcs = [bc_rho, bc_phi]

problem = NonlinearProblem(
    F, u, bcs=bcs, petsc_options_prefix='briz'
)
problem.solve()

rho_sol = u.sub(0).collapse()
phi_sol = u.sub(1).collapse()
+

# 1. Вычисляем разности (rho_ex и phi_ex должны быть UFL-выражениями или Function)
diff_rho = rho_ex - rho_sol
diff_phi = phi_ex - phi_sol

# 2. Формируем интегралы квадратов разностей
# fem.form() компилирует UFL-выражение в готовую к сборке форму
form_rho = form((diff_rho * diff_rho) * ufl.dx)
form_phi = form((diff_phi * diff_phi) * ufl.dx)

# 3. Собираем скалярные значения и берём корень
# assemble_scalar автоматически выполняет MPI-редукцию в параллельных расчётах
err_rho_L2 = np.sqrt(assemble_scalar(form_rho))
err_phi_L2 = np.sqrt(assemble_scalar(form_phi))

print(f"L2 error (rho): {err_rho_L2:.3e}")
print(f"L2 error (phi): {err_phi_L2:.3e}")

