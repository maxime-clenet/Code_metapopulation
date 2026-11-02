"""
Case where the static network at the mean radius predicts extinction but the
seasonal (temporal) network persists.

r(k) = max(0, r_0 + A_half·sin(2π·k·n_cycles/T)), with r_0 < r_p (static extinction).

(a) λ vs A_half at fixed r_0: static ρ(M(r_0)), mean network ρ(M̄), temporal λ_product.
(b) Connectivity radius over time.
(c) Mean occupancy (log scale): static network at r_0 vs seasonal network.
"""

import matplotlib.pyplot as plt
import numpy as np

from metapop import (build_supra, centered_radius, lambda_mean, lambda_product, make_landscape,
                     persistence_radius, simulate, static_lambda, threshold_scenario)
from metapop.plotting import savefig

# === PARAMETERS === #
n, L = 100, 10
e_0, z, beta, c = 0.002, 1.0, 1.0, 1.0
r_0 = 2.2                     # mean radius, below r_p
A_half = 2.5                  # amplitude used for the dynamics panel
T_scan = 500                  # horizon for λ in panel (a)
cycles_scan = [1, 5, 25]      # seasons per T_scan
T_dyn, cycles_dyn = 1000, 10  # horizon and seasons for the dynamics panel
seed = 42

land = make_landscape(n, L, seed)
r_p = persistence_radius(land, c, e_0, z, beta)
lam_static = static_lambda(land, r_0, c, e_0, z, beta)
print(f"r_p = {r_p:.3f},  static λ(r_0 = {r_0}) = {lam_static:.3f}")


def seasonal(A, n_cycles, T):
    return threshold_scenario(land, centered_radius(r_0, A, n_cycles / T), e_0, z, beta)


# === (a) λ vs amplitude === #
A_values = np.linspace(0, 3, 16)
lam_prod = {m: [] for m in cycles_scan}
lam_mean = []
for A in A_values:
    for m in cycles_scan:
        supra = build_supra(seasonal(A, m, T_scan), c, T_scan)
        lam_prod[m].append(lambda_product(supra))
    lam_mean.append(lambda_mean(supra))

# === (c) dynamics === #
P0 = np.full(n, 0.5)
static_scen = threshold_scenario(land, lambda k: r_0, e_0, z, beta)
temporal_scen = seasonal(A_half, cycles_dyn, T_dyn)
P_static = simulate(static_scen, c, T_dyn, P0).mean(axis=0)
P_temporal = simulate(temporal_scen, c, T_dyn, P0).mean(axis=0)
lam_dyn = lambda_product(build_supra(temporal_scen, c, T_dyn))
radius = centered_radius(r_0, A_half, cycles_dyn / T_dyn)
r_t = np.array([radius(k) for k in range(T_dyn + 1)])

# === FIGURE === #
fig = plt.figure(figsize=(14, 7.5))
grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.4], height_ratios=[1, 2.2], hspace=0.08, wspace=0.25)
ax_a = fig.add_subplot(grid[:, 0])
ax_r = fig.add_subplot(grid[0, 1])
ax_p = fig.add_subplot(grid[1, 1], sharex=ax_r)

colors = ['#1b9e77', '#d95f02', '#7570b3']
ax_a.axhline(lam_static, color='k', lw=2, label=f'Static network at $r_0$ = {r_0}')
ax_a.plot(A_values, lam_mean, '--', color='grey', lw=1.5, label=r'Mean network $\rho(\bar M)$')
for m, col in zip(cycles_scan, colors):
    ax_a.plot(A_values, lam_prod[m], '-o', ms=3.5, color=col, lw=1.8,
              label=f'Temporal, {m} season{"s" if m > 1 else ""} / {T_scan} steps')
ax_a.axhline(1, color='crimson', ls=':', lw=1.2)
rescue = A_values[np.array(lam_prod[cycles_scan[0]]) > 1]
if len(rescue):
    ax_a.axvspan(rescue[0], A_values[-1], color='seagreen', alpha=0.1,
                 label='Static: extinction / temporal: persistence')
ax_a.axvline(A_half, color='k', ls=':', lw=1, label=f'$A_{{half}}$ = {A_half} (panel c)')
ax_a.set_xlabel(r'Seasonal amplitude $A_{half}$', fontsize=12)
ax_a.set_ylabel(r'Persistence indicator $\lambda$', fontsize=12)
ax_a.set_title(f'(a)  λ vs amplitude  ($r_0$ = {r_0} < $r_p$ = {r_p:.2f})', fontsize=11)
ax_a.legend(fontsize=8.5, loc='upper left')
ax_a.grid(True, alpha=0.3)

ax_r.plot(r_t, color='steelblue', lw=1.5, label='Seasonal r(k)')
ax_r.axhline(r_0, color='k', lw=1.5, label=f'Static r = {r_0}')
ax_r.axhline(r_p, color='crimson', ls='--', lw=1, label=f'$r_p$ = {r_p:.2f}')
ax_r.set_ylabel('Radius r', fontsize=11)
ax_r.set_ylim(-0.3, r_0 + A_half + 1.6)
ax_r.set_title(f'(b)  Connectivity radius,  $A_{{half}}$ = {A_half}', fontsize=11)
ax_r.legend(fontsize=8, loc='upper right', ncol=3)
ax_r.tick_params(labelbottom=False)
ax_r.grid(True, alpha=0.3)

ax_p.semilogy(P_static, color='k', lw=2, label=f'Static network (λ = {lam_static:.3f}) → extinction')
ax_p.semilogy(P_temporal, color='seagreen', lw=2, label=f'Seasonal network (λ = {lam_dyn:.3f}) → persistence')
ax_p.set_ylim(1e-12, 1.5)
ax_p.set_xlim(0, T_dyn)
ax_p.set_xlabel('Time step k', fontsize=12)
ax_p.set_ylabel(r'Mean occupancy $\langle P \rangle$ (log)', fontsize=11)
ax_p.legend(fontsize=9, loc='lower left')
ax_p.text(0.99, 0.6, '(c)  Mean occupancy', transform=ax_p.transAxes, ha='right', va='top', fontsize=11)
ax_p.grid(True, which='major', alpha=0.3)

savefig('static_vs_temporal.png', fig)
plt.show()
