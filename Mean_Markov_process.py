import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

num_patches = 50  # Number of habitat patches
square_size = 5   # Size of the 2x2 square
c = 1             # Colonization rate
e = 0.01          # Extinction rate
alpha = 0.01 
max_time = 300
n_replicates = 500
dt_sample = 1.0  # Sampling interval for common time grid

# Common time grid for interpolation
common_times = np.arange(0, max_time, dt_sample)
occupancy_matrix = []

for rep in range(n_replicates):
    # --- Initialization per replicate ---
    patch_locations = np.random.rand(num_patches, 2) * square_size
    A = np.random.uniform(1, 4, num_patches)
    A_standardized = A / np.sum(A)
    distances = np.sqrt(((patch_locations[:, None, :] - patch_locations[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    connectivity = np.exp(-alpha * distances)
    X = np.random.choice([0, 1], size=num_patches)
    time = 0
    times = [time]
    occupancies = [np.mean(X)]

    # --- Gillespie simulation ---
    while time < max_time:
        C = c * (X @ connectivity) * (1 - X) * A_standardized
        E = e * X / A_standardized
        lambda_total = np.sum(C) + np.sum(E)
        if lambda_total == 0:
            break
        dt = -np.log(np.random.rand()) / lambda_total
        time += dt
        event_probs = np.concatenate((C, E)) / lambda_total
        event_index = np.random.choice(range(2 * num_patches), p=event_probs)
        if event_index < num_patches:
            X[event_index] = 1
        else:
            X[event_index - num_patches] = 0
        times.append(time)
        occupancies.append(np.mean(X))

    # --- Interpolate to common time grid ---
    interp_func = interp1d(times, occupancies, kind='previous', bounds_error=False, fill_value=(occupancies[0], occupancies[-1]))
    occupancy_interp = interp_func(common_times)
    occupancy_matrix.append(occupancy_interp)

# --- Average occupancy across replicates ---
occupancy_array = np.array(occupancy_matrix)
mean_occupancy = np.mean(occupancy_array, axis=0)
std_occupancy = np.std(occupancy_array, axis=0)

# --- Plot ---
plt.figure(figsize=(8, 4))
plt.plot(common_times, mean_occupancy, label="Mean occupancy", color='green')
plt.fill_between(common_times, mean_occupancy - std_occupancy, mean_occupancy + std_occupancy,
                 color='green', alpha=0.3, label="±1 SD")
plt.xlabel("Time")
plt.ylabel("Proportion of Occupied Patches")
plt.title(f"Smoothed Occupancy Curve over {n_replicates} Simulations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()