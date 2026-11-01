"""
Levins model integrated with an explicit Euler step (dt = 1) on a static landscape:
    P_i(t+1) = P_i + c·Σ_j S_ji P_j·(1 − P_i) − e_i P_i
"""

import matplotlib.pyplot as plt
import numpy as np

from metapop import (connectivity, erdos_renyi_mask, extinction_rates, make_landscape,
                     metapopulation_capacity)

# === PARAMETERS === #
n = 10           # number of habitat patches
L = 2            # side of the spatial domain
c = 1.0          # colonisation rate
e_0 = 0.01       # baseline extinction rate
alpha = 0.1      # dispersal limitation
z = 1            # extinction exponent
T = 200          # number of time steps
p = 1            # Erdős–Rényi edge probability (1 = complete graph)
seed = 42

land = make_landscape(n, L, seed, random_areas=True, periodic=False)
mask = erdos_renyi_mask(n, p, seed)

print(f"Metapopulation capacity (λ_M): {metapopulation_capacity(land, alpha, mask):.4f}")

e = extinction_rates(land.areas, e_0, z)
S = connectivity(land.areas, np.exp(-alpha * land.distances), 1, mask)

P = np.zeros((n, T))
P[:, 0] = np.random.default_rng(seed).random(n)
for t in range(T - 1):
    colonisation = c * (P[:, t] @ S) * (1 - P[:, t])
    P[:, t + 1] = np.clip(P[:, t] + colonisation - e * P[:, t], 0, 1)

plt.figure(figsize=(12, 6))
plt.plot(P.T, lw=1)
plt.xlabel('Time step', fontsize=14)
plt.ylabel('Occupancy probability', fontsize=14)
plt.title('Metapopulation Dynamics with Sparse Spatial Connectivity', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()
