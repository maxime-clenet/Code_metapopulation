"""
Directed seasonal migration between two regions (West, East).

Each region is suitable (e_ok) only in its own season and unsuitable (e_ko) otherwise;
links are directed and change with four phases of equal length:
    summer : links within West      (West suitable, East unsuitable)
    autumn : links West → East      (both regions at e_mig)
    winter : links within East      (East suitable, West unsuitable)
    spring : links East → West      (both regions at e_mig)

'Forward' order follows the seasons above (migrate towards the next suitable region);
'reversed' runs summer → spring → winter → autumn (migrate away from it).
Both orders use exactly the same set of networks, so the static / mean-network and
adiabatic predictions are identical: only the temporal order differs.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from metapop import (build_supra, connectivity, cyclic_schedule, lambda_adiabatic, lambda_mean,
                     lambda_product, make_landscape, simulate)
from metapop.plotting import savefig

# === PARAMETERS === #
n, L = 100, 10
r = 3.0                       # dispersal radius
c = 3.0
e_ok, e_ko, e_mig = 0.02, 0.6, 0.05
T = 1000
periods = [4, 8, 20, 40, 100, 200, 500, 1000]   # one period = four phases
period_dyn = 200
seed = 42

land = make_landscape(n, L, seed, periodic=False)
west = land.positions[:, 0] < L / 2
K = (land.distances < r).astype(float)


def links(src, dst):
    return connectivity(land.areas, K, 1.0, np.outer(src, dst).astype(float))


PHASES = {   # name: (S, e)
    'Summer': (links(west, west), np.where(west, e_ok, e_ko)),
    'Autumn': (links(west, ~west), np.full(n, e_mig)),
    'Winter': (links(~west, ~west), np.where(west, e_ko, e_ok)),
    'Spring': (links(~west, west), np.full(n, e_mig)),
}
ORDERS = {
    'Forward':  ['Summer', 'Autumn', 'Winter', 'Spring'],
    'Reversed': ['Summer', 'Spring', 'Winter', 'Autumn'],
}


def scenario(order, period):
    return cyclic_schedule([PHASES[name] for name in ORDERS[order]], [period // 4] * 4)


# === λ vs season length === #
lam = {o: [lambda_product(build_supra(scenario(o, p), c, T)) for p in periods] for o in ORDERS}
supra_ref = build_supra(scenario('Forward', period_dyn), c, T)
lam_static, lam_adiab = lambda_mean(supra_ref), lambda_adiabatic(supra_ref)
print(f"static / mean network λ = {lam_static:.3f},  adiabatic λ = {lam_adiab:.3f}")
for o in ORDERS:
    print(f"{o:9s}" + "  ".join(f"P={p}:{v:.3f}" for p, v in zip(periods, lam[o])))

# === Dynamics === #
P0 = np.full(n, 0.5)
dyn = {o: simulate(scenario(o, period_dyn), c, T, P0) for o in ORDERS}

# === FIGURE === #
fig = plt.figure(figsize=(16, 9))
grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.22, wspace=0.4)
region_colour = {True: '#1b9e77', False: '#d95f02'}
order_colour = {'Forward': '#1b9e77', 'Reversed': '#7570b3'}

for col, (name, (S, e)) in enumerate(PHASES.items()):
    ax = fig.add_subplot(grid[0, col])
    j, i = np.nonzero(S)            # S[j, i] > 0: j colonises i
    ax.add_collection(LineCollection(np.stack([land.positions[j], land.positions[i]], 1),
                                     colors='grey', lw=0.3, alpha=0.35))
    good = e < 0.5 * (e_ok + e_ko)
    ax.scatter(*land.positions.T, c=[region_colour[w] for w in west], s=30, zorder=3,
               edgecolors=np.where(good, 'k', 'none'), linewidths=1.2,
               alpha=np.where(good | (e == e_mig), 1.0, 0.35))
    ax.axvline(L / 2, color='k', ls=':', lw=0.8)
    if name in ('Autumn', 'Spring'):
        x0, x1 = (0.3, 0.7) if name == 'Autumn' else (0.7, 0.3)
        ax.annotate('', xy=(x1, 1.03), xytext=(x0, 1.03), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', lw=2, color='k'))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    subtitle = {'Summer': 'West suitable', 'Autumn': 'migration W → E',
                'Winter': 'East suitable', 'Spring': 'migration E → W'}[name]
    ax.set_title(f'(a{col + 1})  {name}: {subtitle}', fontsize=10, pad=14)

ax = fig.add_subplot(grid[1, :2])
phase_length = np.array(periods) / 4
ax.axhline(lam_static, color='k', lw=2, label=f'Static / mean network = {lam_static:.3f}')
ax.axhline(lam_adiab, color='k', ls='--', lw=1.2, label=f'Adiabatic (order ignored) = {lam_adiab:.3f}')
for o in ORDERS:
    ax.semilogx(phase_length, lam[o], '-o', color=order_colour[o], lw=2, ms=5,
                label=f'Temporal, {o.lower()} order')
ax.axhline(1, color='crimson', ls=':', lw=1.3)
ax.axvline(period_dyn / 4, color='k', ls=':', lw=0.8)
ax.set_xlabel('Phase length (time steps)', fontsize=12)
ax.set_ylabel(r'$\lambda$', fontsize=12)
ax.set_title('(b)  Same networks, same mean: the order of the seasons decides', fontsize=11)
ax.legend(fontsize=9, loc='center left')
ax.grid(True, which='both', alpha=0.25)

ax = fig.add_subplot(grid[1, 2:])
for o in ORDERS:
    lam_o = lam[o][periods.index(period_dyn)]
    ax.semilogy(dyn[o].mean(axis=0), color=order_colour[o], lw=2,
                label=f'{o} (λ = {lam_o:.3f}) → {"persistence" if lam_o > 1 else "extinction"}')
phase_colours = {'Summer': region_colour[True], 'Winter': region_colour[False]}
for start in range(0, T, period_dyn):
    for q, name in enumerate(ORDERS['Forward']):
        if name in phase_colours:
            a = start + q * period_dyn // 4
            ax.axvspan(a, a + period_dyn // 4, color=phase_colours[name], alpha=0.07, lw=0)
ax.set_ylim(1e-12, 1.5)
ax.set_xlim(0, T)
ax.set_xlabel('Time step k', fontsize=12)
ax.set_ylabel('Mean occupancy (log)', fontsize=11)
ax.set_title(f'(c)  Dynamics, phase length {period_dyn // 4} (shading: forward summer / winter)',
             fontsize=11)
ax.legend(fontsize=9, loc='lower left')
ax.grid(True, alpha=0.25)

savefig('directed_migration.png', fig)
plt.show()
