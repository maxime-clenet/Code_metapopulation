"""
Fixed landscape whose dispersal range shrinks over time: α(k) = α_0·(1 + k/k_scale).
Effect of the horizon T on λ_product.
"""

import matplotlib.pyplot as plt
import numpy as np

from metapop import build_supra, exponential_kernel, lambda_product, make_landscape, make_scenario

# === PARAMETERS === #
n = 10
L = 2
e_0 = 0.01
z = 1
beta = 1
alpha_0 = 0.1
k_scale = 2000
seed = 42
colonisation_rates = np.linspace(0.1, 0.2, 10)
T_values = [50, 200, 2000]

land = make_landscape(n, L, seed, random_areas=True, periodic=False)
scenario = make_scenario(land, e_0, z, beta, exponential_kernel(lambda k: alpha_0 * (1 + k / k_scale)))

plt.figure(figsize=(10, 6))
for T in T_values:
    lambdas = [lambda_product(build_supra(scenario, c, T)) for c in colonisation_rates]
    plt.plot(colonisation_rates, lambdas, marker='o', label=f'T = {T}')

plt.axhline(y=1, color='grey', linestyle='--', label='Threshold λ = 1')
plt.xlabel('Colonisation rate c', fontsize=14)
plt.ylabel('Temporal metapopulation capacity', fontsize=14)
plt.title('Effect of T on the Persistence Threshold', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
