"""
Gap heatmap λ_product(w = 1/T) − λ_static in (r_0, A_half) space,
for r(k) = max(0, r_0 + A_half·sin(2π·k/T)). λ_static = ρ(M(r_0)) is the w = 0 value.
"""

import matplotlib.pyplot as plt
import numpy as np

from metapop import (build_supra, centered_radius, lambda_product, make_landscape,
                     percolation_radius, persistence_radius, static_lambda, threshold_scenario)
from metapop.plotting import savefig

# === PARAMETERS === #
n, L = 100, 10
e_0, z, beta, c = 0.002, 1.0, 1.0, 1.0
T = 500
SEED = 42
w = 1.0 / T

land = make_landscape(n, L, SEED)
r_c = percolation_radius(n, L)
r_p = persistence_radius(land, c, e_0, z, beta)
print(f"r_c = {r_c:.3f},  r_p = {r_p:.3f}\n")

# === GRID === #
r0_arr = np.linspace(0.3, r_p + 2.5, 35)
Ah_arr = np.linspace(0.5, 6.0, 30)

ls_vec = np.array([static_lambda(land, r_0, c, e_0, z, beta) for r_0 in r0_arr])
ls_mat = np.tile(ls_vec, (len(Ah_arr), 1))

lp_mat = np.zeros((len(Ah_arr), len(r0_arr)))
for i, A_h in enumerate(Ah_arr):
    for j, r_0 in enumerate(r0_arr):
        scen = threshold_scenario(land, centered_radius(r_0, A_h, w), e_0, z, beta)
        lp_mat[i, j] = lambda_product(build_supra(scen, c, T))
    print(f"  A_half row {i + 1}/{len(Ah_arr)} done")

# === FIGURE === #
gap = lp_mat - ls_mat
vext = np.max(np.abs(gap))

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.pcolormesh(r0_arr, Ah_arr, gap, cmap='RdBu_r', vmin=-vext, vmax=vext, shading='nearest')
ax.contour(r0_arr, Ah_arr, lp_mat, levels=[1.0], colors='k', linewidths=2.5, linestyles='-')
ax.contour(r0_arr, Ah_arr, ls_mat, levels=[1.0], colors='k', linewidths=1.5, linestyles='--')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r'$\lambda(w{=}1/T) - \lambda(w{=}0)$', fontsize=11)
ax.set_xlabel(r'Mean radius $r_0$', fontsize=13)
ax.set_ylabel(r'Oscillation amplitude $A_{half}$', fontsize=13)
ax.set_title(r'Gap $\lambda(w{=}1/T) - \lambda(w{=}0)$'
             '\n(— temporal threshold,  -- static threshold $r_0 = r_p$)', fontsize=11)

plt.tight_layout()
savefig('fig_gap_phase_diagram.png')
plt.show()
