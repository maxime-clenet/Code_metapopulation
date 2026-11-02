"""
Asynchronous sinks: two habitat types whose quality alternates with the seasons.

Each patch is good (e_good) in one season and bad (e_bad) in the other; type-1 patches
are good in season 1, type-2 patches in season 2. The network itself is fixed (radius r).
All contrasts share the same mean extinction e_mean, hence the same static / mean-network
prediction, but the temporal outcome depends on the contrast and on the season length.

(a) Landscape: habitat types and links.
(b) λ vs season length: static (= mean network), adiabatic, temporal.
(c, d) Occupancy per habitat type for long vs short seasons.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from metapop import (build_supra, connectivity, cyclic_schedule, lambda_adiabatic, lambda_mean,
                     lambda_product, make_landscape, simulate)
from metapop.plotting import savefig

# === PARAMETERS === #
n, L = 100, 10
r = 2.5                                   # fixed connectivity radius
c = 1.0
e_mean = 0.375
contrasts = [0.0, 0.175, 0.325]           # e_good = e_mean − δ, e_bad = e_mean + δ
T = 1000
periods = [2, 4, 8, 20, 40, 100, 200, 500, 1000]   # one period = two seasons
period_slow, period_fast = 200, 4
seed = 42

land = make_landscape(n, L, seed, periodic=False)
habitat = np.random.default_rng(seed).permutation(np.arange(n) % 2)   # 0 or 1, half each
S = connectivity(land.areas, (land.distances < r).astype(float), 1.0)


def scenario(delta, period):
    e_good, e_bad = e_mean - delta, e_mean + delta
    seasons = [np.where(habitat == 0, e_good, e_bad), np.where(habitat == 0, e_bad, e_good)]
    e_of = cyclic_schedule(seasons, [period // 2, period - period // 2])
    return lambda k: (S, e_of(k))


# === λ vs season length === #
lam = {d: [lambda_product(build_supra(scenario(d, p), c, T)) for p in periods] for d in contrasts}
supra_ref = build_supra(scenario(contrasts[-1], period_slow), c, T)
lam_static = lambda_mean(supra_ref)                    # static at mean extinction = mean network
lam_adiab = {d: lambda_adiabatic(build_supra(scenario(d, period_slow), c, T)) for d in contrasts}
print(f"static / mean network λ = {lam_static:.3f}")
for d in contrasts:
    print(f"δ = {d}: adiabatic {lam_adiab[d]:.3f},  temporal "
          + "  ".join(f"P={p}:{v:.3f}" for p, v in zip(periods, lam[d])))

# === Dynamics === #
P0 = np.full(n, 0.5)
dyn = {p: simulate(scenario(contrasts[-1], p), c, T, P0) for p in (period_slow, period_fast)}

# === FIGURE === #
fig = plt.figure(figsize=(15, 9))
grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.6], hspace=0.35, wspace=0.22)
colors_hab = np.array(['#1b9e77', '#d95f02'])
colors_delta = ['#999999', '#7570b3', '#e7298a']

ax = fig.add_subplot(grid[0, 0])
i, j = np.where(np.triu(land.distances < r, 1))
ax.add_collection(LineCollection(np.stack([land.positions[i], land.positions[j]], 1),
                                 colors='grey', lw=0.4, alpha=0.4))
for h, lab in enumerate(['Type 1: good in season 1', 'Type 2: good in season 2']):
    ax.scatter(*land.positions[habitat == h].T, color=colors_hab[h], s=35, edgecolors='k',
               linewidths=0.4, zorder=3, label=lab)
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=1)
ax.set_title(f'(a)  Two interspersed habitat types (r = {r})', fontsize=11)

ax = fig.add_subplot(grid[0, 1])
season_length = np.array(periods) / 2
ax.axhline(lam_static, color='k', lw=2, label=f'Static / mean network (e = {e_mean}) = {lam_static:.3f}')
for d, col in zip(contrasts, colors_delta):
    ax.semilogx(season_length, lam[d], '-o', color=col, lw=2, ms=5,
                label=f'Temporal, e = {e_mean - d:.2f} / {e_mean + d:.2f}')
    if d > 0:
        ax.axhline(lam_adiab[d], color=col, ls='--', lw=1, alpha=0.8)
ax.plot([], [], 'k--', lw=1, label='Adiabatic (same colour)')
ax.axhline(1, color='crimson', ls=':', lw=1.3)
for p, lab in [(period_slow, 'c'), (period_fast, 'd')]:
    ax.axvline(p / 2, color='k', ls=':', lw=0.8)
    ax.text(p / 2, ax.get_ylim()[1], f' ({lab})', va='top', fontsize=9)
ax.set_xlabel('Season length (time steps)', fontsize=12)
ax.set_ylabel(r'$\lambda$', fontsize=12)
ax.set_title('(b)  Same mean landscape, opposite outcomes depending on contrast and season length',
             fontsize=11)
ax.legend(fontsize=8.5, loc='upper left', bbox_to_anchor=(0, 0.93))
ax.grid(True, which='both', alpha=0.25)

for col, (p, lab) in enumerate([(period_slow, 'c'), (period_fast, 'd')]):
    ax = fig.add_subplot(grid[1, col])
    P = dyn[p]
    for h, name in enumerate(['Type 1', 'Type 2']):
        ax.semilogy(P[habitat == h].mean(axis=0), color=colors_hab[h], lw=1.8, label=name)
    ax.semilogy(P.mean(axis=0), color='k', lw=1, ls='--', label='All patches')
    if p >= 20:
        for start in range(0, T, p):
            ax.axvspan(start, start + p // 2, color=colors_hab[0], alpha=0.07, lw=0)
            ax.axvspan(start + p // 2, start + p, color=colors_hab[1], alpha=0.07, lw=0)
    lam_p = lam[contrasts[-1]][periods.index(p)]
    ax.set_ylim(1e-8, 1.5)
    ax.set_xlim(0, T)
    ax.set_xlabel('Time step k', fontsize=11)
    ax.set_ylabel(r'Mean occupancy (log)', fontsize=11)
    ax.set_title(f'({lab})  Season length {p // 2}:  λ = {lam_p:.3f} → '
                 f'{"persistence" if lam_p > 1 else "extinction"}', fontsize=11)
    ax.legend(fontsize=8.5, loc='lower left')
    ax.grid(True, alpha=0.25)

savefig('asynchronous_sinks.png', fig)
plt.show()
