"""Nonlinear discrete-time model on a static landscape, with its persistence threshold ρ(M) = 1."""

import matplotlib.pyplot as plt
import numpy as np

from metapop import (connectivity, erdos_renyi_mask, extinction_rates, make_landscape,
                     metapopulation_capacity, simulate, spectral_radius, transition_matrix)

# === PARAMETERS === #
n = 10           # number of habitat patches
L = 2            # side of the spatial domain
c = 1.0          # colonisation rate
e_0 = 0.01       # baseline extinction rate
z = 1            # extinction exponent
alpha = 0.01     # dispersal limitation
T = 300          # number of time steps
p = 1.0          # Erdős–Rényi edge probability (1 = complete graph)
seed = 42

land = make_landscape(n, L, seed, random_areas=True, periodic=False)
mask = erdos_renyi_mask(n, p, seed)

print(f"Metapopulation capacity (λ_M): {metapopulation_capacity(land, alpha, mask):.4f}")

e = extinction_rates(land.areas, e_0, z)
S = connectivity(land.areas, np.exp(-alpha * land.distances), 1, mask)

P0 = np.random.default_rng(seed).random(n)
P = simulate(lambda k: (S, e), c, T, P0)

rho = spectral_radius(transition_matrix(S, e, c))
print(f"ρ(I − diag(e) + c·Sᵀ) = {rho:.4f}  ({'persistence' if rho > 1 else 'extinction'})")

plt.figure(figsize=(12, 6))
plt.plot(P.T, lw=1)
plt.xlabel('Time step', fontsize=14)
plt.ylabel('Occupancy probability', fontsize=14)
plt.title('Nonlinear Metapopulation Dynamics with Sparse Connectivity', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
