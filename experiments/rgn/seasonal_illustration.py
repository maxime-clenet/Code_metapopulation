"""
Illustration of the metapopulation dynamics on a seasonally changing network.

The connectivity radius follows r(k) = max(0, r_0 + A_half·sin(2π·k·n_seasons/T)):
links appear in the 'wet' season (r large) and break in the 'dry' season (r small).

Figure: four network snapshots over the last cycle (active links, patches coloured by
occupancy) and the time series of r(k) and occupancy. Optionally saves an animation.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from metapop import (build_supra, centered_radius, lambda_product, make_landscape,
                     percolation_radius, persistence_radius, simulate, threshold_scenario)
from metapop.plotting import FIG_DIR, savefig

# === PARAMETERS === #
n, L = 100, 10
e_0, z, beta, c = 0.002, 1.0, 1.0, 1.5
r_0, A_half = 3.0, 2.0        # r(k) oscillates in [r_0 − A_half, r_0 + A_half]
n_seasons = 4                 # number of annual cycles
T = 800
seed = 42
ANIMATE = True
frame_step = 4                # one animation frame every `frame_step` time steps

land = make_landscape(n, L, seed, periodic=False)
radius = centered_radius(r_0, A_half, n_seasons / T)
scenario = threshold_scenario(land, radius, e_0, z, beta)

r_c = percolation_radius(n, L)
r_p = persistence_radius(land, c, e_0, z, beta)
lam = lambda_product(build_supra(scenario, c, T))
print(f"r_c = {r_c:.2f},  r_p = {r_p:.2f},  λ_product = {lam:.3f}")

P = simulate(scenario, c, T, P0=np.random.default_rng(seed).random(n) * 0.5)
r_t = np.array([radius(k) for k in range(T + 1)])
pos = land.positions
cmap, norm = plt.get_cmap('viridis'), plt.Normalize(0, 1)


def active_links(k):
    i, j = np.where(np.triu(land.distances < radius(k), 1))
    return np.stack([pos[i], pos[j]], axis=1)


def draw_network(ax, k):
    ax.add_collection(LineCollection(active_links(k), colors='grey', lw=0.5, alpha=0.5))
    ax.scatter(pos[:, 0], pos[:, 1], c=P[:, k], cmap=cmap, norm=norm,
               s=40, edgecolors='k', linewidths=0.4, zorder=3)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


# Snapshots from the last-but-one wet peak: drying, dry trough, rewetting, then recovery
period = T // n_seasons
k_peak = T - period - 3 * period // 4
k_rewet = k_peak + 3 * period // 4
mean_P = P.mean(axis=0)
k_recovery = k_rewet + int(np.argmax(mean_P[k_rewet:] >= 0.5 * mean_P[k_peak]))
snapshots = {
    'A  wet season':  k_peak,
    'B  drying':      k_peak + period // 4,
    'C  dry season':  k_peak + period // 2,
    'D  rewetting':   k_rewet,
    'E  recovery':    k_recovery,
}

# === FIGURE === #
fig = plt.figure(figsize=(16, 7.5))
grid = fig.add_gridspec(2, len(snapshots), height_ratios=[1, 1], hspace=0.12, wspace=0.08)

for col, (label, k) in enumerate(snapshots.items()):
    ax = fig.add_subplot(grid[0, col])
    draw_network(ax, k)
    ax.set_title(f'{label}\nr = {radius(k):.2f},  ⟨P⟩ = {mean_P[k]:.2f}', fontsize=10)

ax_t = fig.add_subplot(grid[1, :])
ax_t.plot(P.T, color='seagreen', lw=0.3, alpha=0.3)
ax_t.plot(mean_P, color='seagreen', lw=2.5, label='Mean occupancy')
ax_t.set_xlabel('Time step k', fontsize=12)
ax_t.set_ylabel('Occupancy probability', fontsize=12, color='seagreen')
ax_t.set_ylim(0, 1)
ax_t.set_xlim(0, T)

ax_r = ax_t.twinx()
ax_r.plot(r_t, color='steelblue', lw=1.5, label='Connectivity radius r(k)')
ax_r.axhline(r_c, color='steelblue', ls=':', lw=1.2, label=f'$r_c$ = {r_c:.2f} (percolation)')
ax_r.axhline(r_p, color='crimson', ls='--', lw=1.2, label=f'$r_p$ = {r_p:.2f} (static persistence)')
ax_r.set_ylabel('Radius r(k)', fontsize=12, color='steelblue')
ax_r.fill_between(range(T + 1), 0, 1, where=r_t < r_c, transform=ax_r.get_xaxis_transform(),
                  color='orange', alpha=0.12, label='Fragmented network ($r < r_c$)')

for label, k in snapshots.items():
    ax_t.axvline(k, color='k', lw=0.8, alpha=0.6)
    ax_t.text(k, 1.01, label[0], ha='center', va='bottom', fontsize=10, fontweight='bold',
              transform=ax_t.get_xaxis_transform())

handles = ax_t.get_legend_handles_labels()[0][-1:] + ax_r.get_legend_handles_labels()[0]
ax_t.legend(handles=handles, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
ax_t.set_title(f'Seasonal network: $c$ = {c}, $e_i$ = {e_0 * n:.2f},  '
               f'temporal metapopulation capacity λ = {lam:.3f}', fontsize=11, pad=18)

fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes[:len(snapshots)],
             label='Occupancy probability $P_i(k)$', shrink=0.8)
savefig('seasonal_illustration.png', fig)
plt.show()

# === ANIMATION === #
if ANIMATE:
    fig_a, (ax_net, ax_ts) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [1, 1.3]})
    ax_ts.plot(mean_P, color='seagreen', lw=2, label='Mean occupancy')
    ax_ts.plot(r_t / r_t.max(), color='steelblue', lw=1, alpha=0.7, label='r(k) / max r')
    ax_ts.axhline(r_c / r_t.max(), color='steelblue', ls=':', lw=1)
    ax_ts.set_xlim(0, T)
    ax_ts.set_ylim(0, 1.05)
    ax_ts.set_xlabel('Time step k')
    ax_ts.legend(loc='upper right', fontsize=9)
    cursor = ax_ts.axvline(0, color='k', lw=1)
    fig_a.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_net, label='$P_i(k)$', shrink=0.8)

    def update(k):
        ax_net.clear()
        draw_network(ax_net, k)
        season = 'dry, fragmented' if radius(k) < r_c else 'wet' if radius(k) > r_p else 'intermediate'
        ax_net.set_title(f'k = {k},  r = {radius(k):.2f}  ({season})')
        cursor.set_xdata([k, k])

    ani = animation.FuncAnimation(fig_a, update, frames=range(0, T + 1, frame_step), interval=50)
    FIG_DIR.mkdir(exist_ok=True)
    if animation.writers.is_available('ffmpeg'):
        ani.save(FIG_DIR / 'seasonal_illustration.mp4', writer='ffmpeg', fps=20, dpi=120)
    else:
        ani.save(FIG_DIR / 'seasonal_illustration.gif', writer='pillow', fps=20, dpi=80)
    plt.show()
