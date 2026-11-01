"""
Seasonal migration between two groups of patches (first half = winter, second half = summer).
Four phases of equal length, each with its own directed SBM mask (see `two_group_seasonal_masks`).
"""

import matplotlib.pyplot as plt
import numpy as np

from metapop import (Landscape, build_supra, cyclic_mask, exponential_kernel, lambda_product,
                     make_scenario, simulate, two_group_seasonal_masks)
from metapop.plotting import plot_occupancy, plot_persistence_vs_c

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

# Abstract landscape: random symmetric distances in [0, 10], unit areas
rng = np.random.default_rng(seed)
d = np.triu(rng.random((n, n)) * 10, 1)
d = d + d.T
np.fill_diagonal(d, np.inf)
land = Landscape(positions=None, areas=np.ones(n), distances=d)

masks = two_group_seasonal_masks(n, p_11, p_12, p_21, p_22, seed=seed)
scenario = make_scenario(land, e_0, z, beta, exponential_kernel(lambda k: alpha),
                         cyclic_mask(masks, phase_durations))

P0 = np.zeros(n)
P0[:n // 2] = rng.random(n // 2)          # only winter patches initially occupied

lambdas, occupancy = [], []
for c in colonisation_rates:
    lambdas.append(lambda_product(build_supra(scenario, c, T)))
    occupancy.append(simulate(scenario, c, T, P0, floor)[:, -1].mean())

fig, ax = plt.subplots(figsize=(12, 8))
plot_persistence_vs_c(ax, colonisation_rates, lambdas, occupancy, label=f'(T={T})')
ax.set_ylabel('Temporal metapopulation capacity', fontsize=15)
ax.set_title('Impact of Colonisation Rate on Metapopulation Capacity', fontsize=15)
ax.legend(fontsize=12)
plt.show()

winter, summer = np.arange(n // 2), np.arange(n // 2, n)
for c in c_dynamics:
    plot_occupancy(simulate(scenario, c, T, P0, floor), f'Dynamics for c = {c}',
                   groups=[(winter, 'red', 'Winter'), (summer, 'blue', 'Summer')])
