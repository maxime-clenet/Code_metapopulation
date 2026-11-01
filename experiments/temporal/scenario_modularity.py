"""
Modular landscape (SBM mask) whose between-module connectivity decays over time:
p_out(k) = p_in·(1 − k/k_scale).
"""

from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

from metapop import (build_supra, exponential_kernel, lambda_product, make_landscape,
                     make_scenario, sbm_mask, simulate)
from metapop.plotting import plot_persistence_vs_c

# === PARAMETERS === #
n = 30
L = 2
e_0 = 0.01
z = 1
beta = 1
alpha = 0.1
p_in = 0.5
k_scale = 1000
seed = 42
colonisation_rates = np.linspace(0.6, 1, 10)
T_values = [100, 500]

n_modules = int(np.sqrt(n))
sizes = [n // n_modules] * n_modules


@lru_cache(maxsize=None)
def modular_mask(k):
    p_out = p_in * (1 - k / k_scale)
    p_matrix = np.full((n_modules, n_modules), p_out)
    np.fill_diagonal(p_matrix, p_in)
    return sbm_mask(sizes, p_matrix, seed=seed)


land = make_landscape(n, L, seed, periodic=False)
scenario = make_scenario(land, e_0, z, beta, exponential_kernel(lambda k: alpha), modular_mask)
P0 = np.random.default_rng(seed).random(n)

fig, ax = plt.subplots(figsize=(12, 8))
for T in T_values:
    lambdas, occupancy = [], []
    for c in colonisation_rates:
        lambdas.append(lambda_product(build_supra(scenario, c, T)))
        occupancy.append(simulate(scenario, c, T, P0)[:, -1].mean())
    plot_persistence_vs_c(ax, colonisation_rates, lambdas, occupancy, label=f'(T={T})')

ax.set_ylabel('Value')
ax.set_title('Temporal Metapopulation Capacity and Final Occupancy vs. Colonisation Rate')
ax.legend()
plt.show()
