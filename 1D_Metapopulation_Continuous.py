import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# === PARAMETERS === #
num_patches = 10           # Number of habitat patches
square_size = 2            # Size of the spatial domain (2x2 square)
c = 1.0                    # Colonization rate
e_0 = 0.01                 # Baseline extinction rate
alpha = 0.1                # Dispersal limitation coefficient
z = 1                      # Extinction exponent
max_time = 200             # Number of time steps
p = 1                   # Edge probability for Erdős-Rényi connectivity
seed = 42                  # Random seed

# === INITIALIZATION === #
np.random.seed(seed)

# Generate patch locations and areas
patch_locations = np.random.rand(num_patches, 2) * square_size
A = np.random.uniform(1, 4, num_patches)
A_std = A / np.sum(A)

# Compute Euclidean distances
distances = np.linalg.norm(patch_locations[:, None, :] - patch_locations[None, :, :], axis=2)
np.fill_diagonal(distances, np.inf)

# Generate Erdős-Rényi graph and adjacency mask
G = nx.erdos_renyi_graph(num_patches, p, seed=seed)
adj_matrix = nx.to_numpy_array(G)
np.fill_diagonal(adj_matrix, 0)

# === LANDSCAPE MATRIX (for capacity, sparsified) === #
A_i = A_std[:, None]
A_j = A_std[None, :]
landscape_matrix = A_i * A_j * np.exp(-alpha * distances)
np.fill_diagonal(landscape_matrix, 0)
landscape_matrix *= adj_matrix  # Apply sparsity constraint

# Metapopulation capacity (leading eigenvalue)
lambda_M = np.max(np.linalg.eigvals(landscape_matrix).real)
print(f"Metapopulation capacity (λ_M): {lambda_M:.4f}")

# === EXTINCTION RATES === #
extinction_rates = e_0 * A_std ** (-z)

# === CONNECTIVITY MATRIX (used in dynamics) === #
S = A_std[None, :] * np.exp(-alpha * distances)
np.fill_diagonal(S, 0)
S *= adj_matrix  # Apply same graph constraint

# === SIMULATION === #
P = np.zeros((num_patches, max_time))
P[:, 0] = np.random.rand(num_patches)

for t in range(max_time - 1):
    for i in range(num_patches):
        colonization = c * np.sum(P[:, t] * S[:, i]) * (1 - P[i, t])
        extinction = extinction_rates[i] * P[i, t]
        P[i, t + 1] = np.clip(P[i, t] + colonization - extinction, 0, 1)

# === PLOTTING === #
plt.figure(figsize=(12, 6))
for i in range(num_patches):
    plt.plot(P[i, :], lw=1)

plt.xlabel('Time step', fontsize=14)
plt.ylabel('Occupancy probability', fontsize=14)
plt.title('Metapopulation Dynamics with Sparse Spatial Connectivity', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()