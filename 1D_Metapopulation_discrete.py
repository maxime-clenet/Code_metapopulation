import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# === PARAMETERS === #
num_patches = 10           # Number of habitat patches
square_size = 2            # Spatial domain size (2x2 square)
c = 1.0                    # Colonization rate
e_0 = 0.01                # Baseline extinction rate
z = 1                      # Extinction exponent
alpha = 0.01               # Dispersal limitation coefficient
T = 300                    # Number of time steps
p = 1.0                    # Erdős-Rényi edge probability (fully connected if 1)
seed = 42                  # Random seed for reproducibility

# === INITIALIZATION === #
np.random.seed(seed)

# Generate spatial patch locations and standardized areas
patch_locations = np.random.rand(num_patches, 2) * square_size
A = np.random.uniform(1, 4, num_patches)
A_std = A / np.sum(A)

# Compute Euclidean distances
distances = np.linalg.norm(patch_locations[:, None, :] - patch_locations[None, :, :], axis=2)
np.fill_diagonal(distances, np.inf)  # Prevent self-colonization

# Generate Erdős-Rényi graph and adjacency mask
G = nx.erdos_renyi_graph(num_patches, p, seed=seed)
adj_matrix = nx.to_numpy_array(G)
np.fill_diagonal(adj_matrix, 0)

# === LANDSCAPE MATRIX (for capacity) with sparsity === #
A_i = A_std[:, None]
A_j = A_std[None, :]
landscape_matrix = A_i * A_j * np.exp(-alpha * distances)
np.fill_diagonal(landscape_matrix, 0)
landscape_matrix *= adj_matrix  # Apply sparsity

# Metapopulation capacity
lambda_M = np.max(np.linalg.eigvals(landscape_matrix).real)
print(f"Metapopulation capacity (λ_M): {lambda_M:.4f}")

# === EXTINCTION RATES === #
e = e_0 * A_std ** (-z)

# === CONNECTIVITY MATRIX (used in nonlinear dynamics) === #
# S_ji = A_j * exp(-alpha * d_ji)
S = A_std[None, :] * np.exp(-alpha * distances)
np.fill_diagonal(S, 0)
S *= adj_matrix  # Apply same sparsity constraint

# === SIMULATION (Nonlinear colonization model) === #
P = np.zeros((num_patches, T))
P[:, 0] = np.random.rand(num_patches)

for k in range(T - 1):
    q = np.ones(num_patches)
    for i in range(num_patches):
        for j in range(num_patches):
            q[i] *= (1 - c * S[j, i] * P[j, k])
    P[:, k + 1] = 1 - (1 - (1 - e) * P[:, k]) * q
    P[:, k + 1] = np.clip(P[:, k + 1], 0, 1)

# === CONNECTIVITY MATRIX LEADING EIGENVALUE === #
lambda_max = np.max(np.linalg.eigvals(c * S - np.diag(e)).real)
print(f"Max eigenvalue of the connectivity matrix S: {lambda_max:.4f}")

# === PLOT === #
plt.figure(figsize=(12, 6))
for i in range(num_patches):
    plt.plot(P[i, :], lw=1)

plt.xlabel('Time step', fontsize=14)
plt.ylabel('Occupancy probability', fontsize=14)
plt.title('Nonlinear Metapopulation Dynamics with Sparse Connectivity', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()