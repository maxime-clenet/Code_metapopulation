"""
Figures 4 and 5 of the paper (seasonal random geometric network), 16:9, black only.
The amplitude is chosen so the seasonal network is never fully disconnected nor fully
connected: r(k) ∈ [0.95, 3.95], i.e. 3 to 48 neighbours per patch on average (out of 99).

Figure 4: static network (left) vs seasonal network at three moments of the cycle (right).
Figure 5: mean occupancy over time, static vs seasonal.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from metapop import (build_supra, centered_radius, lambda_product, make_landscape, simulate,
                     static_lambda, threshold_scenario)
from metapop.plotting import savefig

# === PARAMETERS === #
n, L = 100, 10
e_0, z, beta, c = 0.002, 1.0, 1.0, 1.0
r_0, A_half = 2.45, 1.5
T, n_cycles = 1500, 15
seed = 42

STATIC = TEMPORAL = 'black'
plt.rcParams.update({'font.size': 16, 'axes.spines.top': False, 'axes.spines.right': False})

land = make_landscape(n, L, seed)
radius = centered_radius(r_0, A_half, n_cycles / T)
period = T // n_cycles
static_scen = threshold_scenario(land, lambda k: r_0, e_0, z, beta)
temporal_scen = threshold_scenario(land, radius, e_0, z, beta)

lam_static = static_lambda(land, r_0, c, e_0, z, beta)
lam_temporal = lambda_product(build_supra(temporal_scen, c, T))
P0 = np.full(n, 0.5)
P_static = simulate(static_scen, c, T, P0).mean(axis=0)
P_temporal = simulate(temporal_scen, c, T, P0).mean(axis=0)


def draw_network(ax, r, colour):
    # links drawn without the torus wrap-around, for readability
    d = np.linalg.norm(land.positions[:, None] - land.positions[None], axis=2)
    i, j = np.where(np.triu(d < r, 1))
    ax.add_collection(LineCollection(np.stack([land.positions[i], land.positions[j]], 1),
                                     colors=colour, lw=0.6, alpha=0.45))
    ax.scatter(*land.positions.T, s=28, color=colour, edgecolors='white', linewidths=0.6, zorder=3)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.axis('off')


# === FIGURE 4: networks === #
moments = [('Wet season', period // 4), ('Transition', period // 2), ('Dry season', 3 * period // 4)]
k_axis = np.arange(period + 1)

fig = plt.figure(figsize=(16, 9))
grid = fig.add_gridspec(2, 5, width_ratios=[1, 0.3, 1, 1, 1], height_ratios=[1, 0.5],
                        hspace=0.12, wspace=0.08, left=0.05, right=0.98, top=0.82, bottom=0.08)

ax = fig.add_subplot(grid[0, 0])
draw_network(ax, r_0, STATIC)
ax.set_title(f'r = {r_0}', fontsize=15, color=STATIC)
for col, (name, k) in enumerate(moments):
    ax = fig.add_subplot(grid[0, 2 + col])
    draw_network(ax, radius(k), TEMPORAL)
    ax.set_title(f'{name}\nr = {radius(k):.2f}', fontsize=15, color=TEMPORAL)

ax = fig.add_subplot(grid[1, 0])
ax.plot(k_axis, np.full_like(k_axis, r_0, dtype=float), color=STATIC, lw=3)
ax_t = fig.add_subplot(grid[1, 2:], sharey=ax)
ax_t.plot(k_axis, [radius(k) for k in k_axis], color=TEMPORAL, lw=3)
ax_t.axhline(r_0, color=STATIC, ls=':', lw=1.5)
for name, k in moments:
    ax_t.plot(k, radius(k), 'o', color=TEMPORAL, ms=10, mec='white', mew=1.5)
for a in (ax, ax_t):
    a.set_xticks([])
    a.set_xlabel('one year', fontsize=14)
ax.set_ylabel('radius r', fontsize=14)
ax_t.tick_params(labelleft=False)

fig.text(0.205, 0.93, 'Static network', ha='center', fontsize=24, fontweight='bold', color=STATIC)
fig.text(0.695, 0.93, 'Seasonal network', ha='center', fontsize=24, fontweight='bold', color=TEMPORAL)
fig.text(0.695, 0.885, f'same mean radius r = {r_0}, links come and go with the seasons',
         ha='center', fontsize=15, color=TEMPORAL)
savefig('Figure 4.png', fig)
plt.show()

# === FIGURE 5: outcome === #
fig, ax = plt.subplots(figsize=(16, 9))
fig.subplots_adjust(left=0.1, right=0.78, top=0.88, bottom=0.12)
ax.semilogy(P_static, color=STATIC, lw=4, ls='--')
ax.semilogy(P_temporal, color=TEMPORAL, lw=4)
ax.set_ylim(min(P_static.min(), P_temporal.min()) / 5, 2)
ax.set_xlim(0, T)
ax.set_xlabel('Time', fontsize=20)
ax.set_ylabel('Fraction of occupied patches (log)', fontsize=20)
ax.set_xticks([])

ax.text(T * 1.01, np.sqrt(P_temporal[-period:].max() * P_temporal[-period:].min()),
        f'Seasonal network\nλ = {lam_temporal:.3f}  →  persists',
        color=TEMPORAL, fontsize=19, fontweight='bold', va='center')
ax.text(T * 1.01, P_static[-1], f'Static network\nλ = {lam_static:.3f}  →  goes extinct',
        color=STATIC, fontsize=19, fontweight='bold', va='center')
ax.set_title('Same mean connectivity, opposite outcomes', fontsize=26, fontweight='bold', pad=20)
savefig('Figure 5.png', fig)
plt.show()
