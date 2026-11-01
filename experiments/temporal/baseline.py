"""Oscillating threshold network r(k) = (A/2)(1 + sin 2πwk): λ_product vs final occupancy as c varies."""

import matplotlib.pyplot as plt
import numpy as np

from metapop import (build_supra, lambda_product, make_landscape, simulate, sinusoidal_radius,
                     threshold_scenario)
from metapop.plotting import plot_occupancy, plot_persistence_vs_c

# === PARAMETERS === #
n = 50
L = 2
e_0 = 0.01
z = 1
beta = 1
T = 1000
A, w = 4.0, 0.005
seed = 42
colonisation_rates = np.linspace(0.1, 0.2, 100)

land = make_landscape(n, L, seed, random_areas=True, periodic=False)
scenario = threshold_scenario(land, sinusoidal_radius(A, w), e_0, z, beta)
P0 = np.random.default_rng(seed).random(n)

lambdas, occupancy = [], []
for c in colonisation_rates:
    lambdas.append(lambda_product(build_supra(scenario, c, T)))
    P = simulate(scenario, c, T, P0)
    occupancy.append(P[:, -1].mean())

fig, ax = plt.subplots(figsize=(10, 6))
plot_persistence_vs_c(ax, colonisation_rates, lambdas, occupancy)
ax.set_ylabel('Value', fontsize=14)
ax.set_title('Stability and Persistence vs. Colonisation Rate', fontsize=16)
ax.legend()
plt.tight_layout()
plt.show()

plot_occupancy(P, f'Occupancy dynamics for c = {colonisation_rates[-1]:.2f}')
