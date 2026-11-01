"""Mean occupancy of the stochastic Levins model over many replicates (new random landscape each time)."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from metapop import make_landscape
from metapop.stochastic import gillespie

# === PARAMETERS === #
n = 50           # number of habitat patches
L = 5            # side of the spatial domain
c = 1            # colonisation rate
e = 0.01         # extinction rate
alpha = 0.01     # dispersal limitation
max_time = 300
n_replicates = 500
dt_sample = 1.0  # sampling interval of the common time grid
seed = 0

rng = np.random.default_rng(seed)
grid = np.arange(0, max_time, dt_sample)
occupancy = np.zeros((n_replicates, len(grid)))

for rep in range(n_replicates):
    land = make_landscape(n, L, seed=rng.integers(2**31), random_areas=True, periodic=False)
    X0 = rng.choice([0, 1], size=n)
    times, states = gillespie(X0, np.exp(-alpha * land.distances), land.areas, c, e, max_time, rng=rng)
    occ = states.mean(axis=1)
    occupancy[rep] = interp1d(times, occ, kind='previous', bounds_error=False,
                              fill_value=(occ[0], occ[-1]))(grid)

mean, std = occupancy.mean(axis=0), occupancy.std(axis=0)

plt.figure(figsize=(8, 4))
plt.plot(grid, mean, label="Mean occupancy", color='green')
plt.fill_between(grid, mean - std, mean + std, color='green', alpha=0.3, label="±1 SD")
plt.xlabel("Time")
plt.ylabel("Proportion of Occupied Patches")
plt.title(f"Mean Occupancy over {n_replicates} Simulations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
