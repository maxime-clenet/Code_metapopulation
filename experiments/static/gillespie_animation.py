"""Single Gillespie realisation of the stochastic Levins model, saved as an MP4 animation."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from metapop import make_landscape, metapopulation_capacity
from metapop.plotting import FIG_DIR
from metapop.stochastic import gillespie

# === PARAMETERS === #
n = 50           # number of habitat patches
L = 5            # side of the spatial domain
c = 1            # colonisation rate
e = 0.015        # extinction rate
alpha = 0.01     # dispersal limitation
max_time = 100
max_frames = 1000
seed = 42

land = make_landscape(n, L, seed, random_areas=True, periodic=False)
lambda_M = metapopulation_capacity(land, alpha)
print(f"Metapopulation capacity (λ_M): {lambda_M:.4f}")

rng = np.random.default_rng(seed)
X0 = rng.choice([0, 1], size=n)
times, states = gillespie(X0, np.exp(-alpha * land.distances), land.areas, c, e,
                          max_time, max_events=max_frames - 1, rng=rng)

# === ANIMATION === #
fig, ax = plt.subplots(figsize=(6, 6))


def update(frame):
    ax.clear()
    colors = np.where(states[frame] == 1, 'blue', 'red')     # blue = occupied, red = empty
    ax.scatter(land.positions[:, 0], land.positions[:, 1], c=colors, s=1000 * land.areas)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Metapopulation Dynamics (Time: {times[frame]:.2f})\n"
                 f"Metapopulation Capacity: {lambda_M:.2f}")
    ax.grid(True)


ani = animation.FuncAnimation(fig, update, frames=len(states), interval=200, blit=False)
FIG_DIR.mkdir(exist_ok=True)
ani.save(FIG_DIR / "metapopulation_animation.mp4", writer='ffmpeg', fps=5)
plt.show()

# === OCCUPANCY === #
plt.figure(figsize=(8, 4))
plt.plot(times, states.mean(axis=1), color='green')
plt.xlabel("Time")
plt.ylabel("Proportion of Occupied Patches")
plt.title("Occupancy Dynamics Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
