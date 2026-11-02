"""
Seasonal two-group landscape with unequal phases (200, 50, 200, 50):
λ_product vs final occupancy vs the worst instantaneous λ, as c varies.
"""

import matplotlib.pyplot as plt
import numpy as np

from metapop import (Landscape, build_supra, cyclic_schedule, exponential_kernel, lambda_product,
                     make_scenario, min_instantaneous_lambda, simulate, two_group_seasonal_masks)
from metapop.plotting import plot_persistence_vs_c

# === PARAMETERS === #
n = 200
alpha = 3
e_0 = 0.1
z = 1.0
beta = 1.0
p_11, p_12, p_21, p_22 = 0.6, 0.02, 0.2, 0.7
phase_durations = [200, 50, 200, 50]
floor = 1e-3              # occupancies below this are set to 0
T = 1000
seed = 42
colonisation_rates = np.linspace(0.05, 0.7, 30)

# Abstract landscape: random symmetric distances in [0, 10], unit areas
rng = np.random.default_rng(seed)
d = np.triu(rng.random((n, n)) * 10, 1)
d = d + d.T
np.fill_diagonal(d, np.inf)
land = Landscape(positions=None, areas=np.ones(n), distances=d)

masks = two_group_seasonal_masks(n, p_11, p_12, p_21, p_22, seed=seed)
scenario = make_scenario(land, e_0, z, beta, exponential_kernel(lambda k: alpha),
                         cyclic_schedule(masks, phase_durations))
P0 = rng.random(n)

lambdas, occupancy, worst = [], [], []
for c in colonisation_rates:
    supra = build_supra(scenario, c, T)
    lambdas.append(lambda_product(supra))
    worst.append(min_instantaneous_lambda(supra))
    occupancy.append(simulate(scenario, c, T, P0, floor)[:, -1].mean())

fig, ax = plt.subplots(figsize=(12, 8))
plot_persistence_vs_c(ax, colonisation_rates, lambdas, occupancy, label=f'(T={T})')
ax.plot(colonisation_rates, worst, marker='s', ls='-.', label=r'$\min_k \rho(M_k)$')
ax.set_ylabel('Value')
ax.set_title('Temporal capacity, final occupancy and worst instantaneous ρ vs. colonisation rate')
ax.legend()
plt.show()
