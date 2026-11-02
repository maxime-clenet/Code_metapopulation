"""
Seasonal migration between two groups of patches (first half = winter, second half = summer).
Four phases of equal length, each with its own directed SBM mask (see `two_group_seasonal_masks`);
with p_11 = p_22 = 0 the cycle is: rest, migration winter → summer, rest, migration summer → winter.

Single figure:
top: temporal capacity λ and mean occupancy over the last year vs colonisation rate c;
bottom: occupancy of winter and summer patches below (left) and above (right) the threshold,
linked by arrows to the corresponding c in the top panel.
"""

import matplotlib.pyplot as plt
import numpy as np

from metapop import (Landscape, build_supra, cyclic_schedule, exponential_kernel, lambda_product,
                     make_scenario, simulate, two_group_seasonal_masks)
from metapop import threshold_crossing
from matplotlib.patches import ConnectionPatch

from metapop.plotting import savefig

# === PARAMETERS === #
n = 50
alpha = 1
e_0 = 0.04
z = 1.0
beta = 1.0
p_11, p_12, p_21, p_22 = 0, 1, 1, 0
phase_durations = [40, 40, 40, 40]
floor = 1e-3              # occupancies below this are set to 0
T = 10000
seed = 42
colonisation_rates = np.linspace(0.02, 0.5, 50)
c_dynamics = [0.2, 0.4]
windows = [(0, 500), (0, 500)]   # time windows shown in the bottom panels

# Abstract landscape: random symmetric distances in [0, 10], unit areas
rng = np.random.default_rng(seed)
d = np.triu(rng.random((n, n)) * 10, 1)
d = d + d.T
np.fill_diagonal(d, np.inf)
land = Landscape(positions=None, areas=np.ones(n), distances=d)

masks = two_group_seasonal_masks(n, p_11, p_12, p_21, p_22, seed=seed)
scenario = make_scenario(land, e_0, z, beta, exponential_kernel(lambda k: alpha),
                         cyclic_schedule(masks, phase_durations))

P0 = np.zeros(n)
P0[:n // 2] = rng.random(n // 2)          # only winter patches initially occupied

period = sum(phase_durations)
lambdas, occupancy = [], []
for c in colonisation_rates:
    lambdas.append(lambda_product(build_supra(scenario, c, T)))
    occupancy.append(simulate(scenario, c, T, P0, floor)[:, -period:].mean())
c_star = threshold_crossing(colonisation_rates, lambdas)
print(f"λ = 1 at c ≈ {c_star:.3f}")

# === FIGURE === #
WINTER, SUMMER, LAMBDA, OCC = '#2b83ba', '#f28e2b', 'black', '#7f7f7f'
groups = [(np.arange(n // 2), WINTER, 'Winter patches'), (np.arange(n // 2, n), SUMMER, 'Summer patches')]
plt.rcParams.update({'axes.spines.top': False, 'font.size': 15,
                     'axes.labelsize': 17, 'xtick.labelsize': 15, 'ytick.labelsize': 15})

fig = plt.figure(figsize=(15, 10.5))
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.5, wspace=0.18)

# top: threshold
ax = ax_top = fig.add_subplot(grid[0, :])
ax.axvspan(colonisation_rates[0], c_star, color='0.92', lw=0)
ax.axhline(1, color=LAMBDA, ls=':', lw=1)
ax.axvline(c_star, color=LAMBDA, ls='--', lw=1)
ax.plot(colonisation_rates, lambdas, '-o', color=LAMBDA, ms=5, lw=2.5)
ax.set_xlabel('Colonisation rate $c$')
ax.set_ylabel(r'Temporal capacity $\lambda$')
ax.set_xlim(colonisation_rates[0], colonisation_rates[-1])
ax_o = ax.twinx()
ax_o.plot(colonisation_rates, occupancy, '-s', color=OCC, ms=5, lw=2.5)
ax_o.set_ylabel('Mean occupancy', color=OCC)
ax_o.tick_params(axis='y', colors=OCC)
ax_o.set_ylim(0, None)
ax.plot([], [], '-o', color=LAMBDA, label=r'Temporal capacity $\lambda$')
ax.plot([], [], '-s', color=OCC, label='Mean occupancy (simulation)')
ax.legend(loc='upper left', fontsize=15, frameon=False)
ax.text((colonisation_rates[0] + c_star) / 2, ax.get_ylim()[0], 'extinction\n', ha='center',
        va='bottom', color='0.4', fontsize=16)
ax.text((c_star + colonisation_rates[-1]) / 2, ax.get_ylim()[0], 'persistence\n', ha='center',
        va='bottom', color='0.4', fontsize=16)
for c in c_dynamics:
    ax.plot(c, np.interp(c, colonisation_rates, lambdas), 'o', ms=14, mfc='none', mec='crimson', mew=2.5)

# bottom: mean occupancy per group, each pointed to by an arrow from its c in the top panel
for col, (c, (t0, t1)) in enumerate(zip(c_dynamics, windows)):
    ax = fig.add_subplot(grid[1, col])
    P = simulate(scenario, c, T, P0, floor)
    t = np.arange(T + 1)
    for start in range(t0 - t0 % period, t1, period):
        for q, label in ((1, 'W → S'), (3, 'S → W')):
            a0 = start + q * phase_durations[0]
            if t0 <= a0 < t1:
                ax.axvspan(a0, a0 + phase_durations[q], color='0.9', lw=0)
                if (t1 - t0) <= 4 * period:
                    ax.text(a0 + phase_durations[q] / 2, 0.6, label, ha='center', va='top',
                            fontsize=12, color='0.35')
    for idx, colour, name in groups:
        ax.plot(t[t0:t1], P[idx, t0:t1].mean(axis=0), color=colour, lw=3, label=name)
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 0.62)
    ax.set_xlabel('Time step')
    if col == 0:
        ax.set_ylabel('Mean occupancy')
    fig.add_artist(ConnectionPatch(
        xyA=(c, np.interp(c, colonisation_rates, lambdas)), coordsA=ax_top.transData,
        xyB=(0.5, 1.03), coordsB=ax.transAxes, arrowstyle='-|>', mutation_scale=22,
        color='crimson', lw=2.2, shrinkA=11, shrinkB=0))

top_bottom_row = ax.get_position().y1
fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=2, fontsize=16,
           frameon=False, bbox_to_anchor=(0.5, top_bottom_row + 0.005))

savefig('scenario_season.png', fig)
plt.show()
