import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Parameters
num_patches = 50  # Number of habitat patches
square_size = 5   # Size of the 2x2 square
c = 1             # Colonization rate
e = 0.015          # Extinction rate
alpha = 0.01        # Dispersal limitation factor
max_time = 100   # Maximum simulation time (increased to allow convergence)
frames = 1000     # Number of frames for animation

# Generate random habitat patch locations
np.random.seed(42)
patch_locations = np.random.rand(num_patches, 2) * square_size

# Step 1: Sample site sizes from a uniform distribution between 0 and 4
A = np.random.uniform(1, 4, num_patches)

# Step 2: Standardize site sizes so that the sum equals 1
A_standardized = A / np.sum(A)

# Compute distance matrix
distances = np.sqrt(((patch_locations[:, None, :] - patch_locations[None, :, :]) ** 2).sum(axis=2))
np.fill_diagonal(distances, np.inf)  # No self-colonization

# Connectivity matrix: S_ji = exp(-alpha * d_ji)
connectivity = np.exp(-alpha * distances)

# Step 3: Compute the landscape matrix M_ij = A_i * A_j * exp(-alpha * d_ij)

A_i = A_standardized[:, None]  # shape (n, 1)
A_j = A_standardized[None, :]  # shape (1, n)
landscape_matrix = A_i * A_j * np.exp(-alpha * distances)

# Step 4: Set diagonal to zero (M_ii = 0)
np.fill_diagonal(landscape_matrix, 0)

# Step 5: Compute metapopulation capacity (leading eigenvalue)
lambda_M = np.linalg.eigvals(landscape_matrix).max().real

# Initialize patch states (randomly occupied)
X = np.random.choice([0, 1], size=num_patches)

# Initialize simulation variables
time = 0
times = [time]
state_history = [X.copy()]

# Gillespie Simulation Loop until max_time is reached
while time < max_time and len(state_history) < frames:
    # Compute colonization and extinction rates
    C = c * (X @ connectivity) * (1 - X) * A_standardized  # Colonization depends on patch area
    E = e * X / A_standardized  # Extinction inversely depends on patch area

    # Total event rate
    lambda_total = np.sum(C) + np.sum(E)
    if lambda_total == 0:
        break  # Stop if no more events possible

    # Sample time until next event
    dt = -np.log(np.random.rand()) / lambda_total
    time += dt

    # Choose which event occurs
    event_probs = np.concatenate((C, E)) / lambda_total
    event_index = np.random.choice(range(2 * num_patches), p=event_probs)

    # Apply the event
    if event_index < num_patches:
        X[event_index] = 1  # Colonization
    else:
        X[event_index - num_patches] = 0  # Extinction

    # Store results
    times.append(time)
    state_history.append(X.copy())

# Create figure for animation
fig, ax = plt.subplots(figsize=(6, 6))
def update(frame):
    ax.clear()
    colors = np.array(["blue" if state == 1 else "red" for state in state_history[frame]])  # Blue for occupied, Red for empty
    ax.scatter(patch_locations[:, 0], patch_locations[:, 1], c=colors, s=1000 * A_standardized)  # Multiply by 1000 for better visualization
    ax.set_xlim(0, square_size)
    ax.set_ylim(0, square_size)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Metapopulation Dynamics (Time: {times[frame]:.2f})\nMetapopulation Capacity: {lambda_M:.2f}")
    plt.grid(True)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(state_history), interval=200, blit=False)

# Save animation as MP4
save_path = os.path.join(os.path.expanduser("~"), "Desktop", "simu", "metapopulation_animation.mp4")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
ani.save(save_path, writer='ffmpeg', fps=5)

plt.show()

# Return the metapopulation capacity
print(f"Metapopulation Capacity (lambda_M): {lambda_M}")

# Compute occupancy (mean of 1s in X at each time point)
occupancy = [np.mean(state) for state in state_history]

# Create the plot
plt.figure(figsize=(8, 4))
plt.plot(times, occupancy, color='green')
plt.xlabel("Time")
plt.ylabel("Proportion of Occupied Patches")
plt.title("Occupancy Dynamics Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()