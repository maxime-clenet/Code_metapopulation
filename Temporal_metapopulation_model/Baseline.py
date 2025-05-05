import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_matrices(n, c, e_0, z, beta, T, scenario_func):
    np.random.seed(42)

    # Initialize occupancy probability matrix
    P = np.zeros((n, T))
    P[:, 0] = np.random.random(n)

    # Store time-varying transition matrices
    supra_adjacency_matrix = np.zeros((T-1, n, n))

    for k in range(T-1):
        S, e = scenario_func(n, e_0, z, beta)
        supra_adjacency_matrix[k] = np.eye(n) - np.diag(e) + c * S

        # Nonlinear update
        q = np.ones(n)
        for i in range(n):
            for j in range(n):
                q[i] *= (1 - c * S[j, i] * P[j, k])
        P[:, k + 1] = 1 - (1 - (1 - e) * P[:, k]) * q
        P[:, k + 1] = np.clip(P[:, k + 1], 0, 1)

    return P, supra_adjacency_matrix

def create_scenario_1(n, e_0, z, beta, alpha=0.01, square_size=2, p=1):
    # Generate spatial locations and areas
    patch_locations = np.random.rand(n, 2) * square_size
    A = np.random.uniform(1, 4, n)
    A_std = A / np.sum(A)

    # Compute pairwise distances
    distances = np.linalg.norm(patch_locations[:, None, :] - patch_locations[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)

    # Extinction rates
    e = e_0 * A_std ** (-z)

    # Build connectivity matrix: S_ji = A_j^beta * exp(-alpha * d_ji)
    A_j = A_std[None, :] ** beta
    S = A_j * np.exp(-alpha * distances)
    np.fill_diagonal(S, 0)

    # Apply sparsity from Erdős-Rényi graph
    G = nx.erdos_renyi_graph(n, p, seed=42)
    adjacency_matrix = nx.to_numpy_array(G)
    np.fill_diagonal(adjacency_matrix, 0)
    S *= adjacency_matrix

    return S, e

def max_eigenvalue_product_matrix(supra_adjacency_matrix):
    n = supra_adjacency_matrix.shape[1]
    T_minus_1 = supra_adjacency_matrix.shape[0]

    product_matrix = np.eye(n)
    for k in range(T_minus_1):
        product_matrix = np.dot(supra_adjacency_matrix[k], product_matrix)

    eigenvalues = np.linalg.eigvals(product_matrix)**(1/T_minus_1)
    return np.max(eigenvalues.real)

# === Parameters === #
n = 10
e_0 = 0.01
z = 1
beta = 1
T = 1000
colonization_rates = np.linspace(0.1, 0.2, 100)

# === Run simulation over varying colonization rates === #
max_eigenvalues = []
mean_final_probs = []

for c in colonization_rates:
    P, supra = generate_matrices(n, c, e_0, z, beta, T, create_scenario_1)
    max_eigenvalue = max_eigenvalue_product_matrix(supra)
    max_eigenvalues.append(max_eigenvalue)
    mean_final_probs.append(np.mean(P[:, -1]))

# === Plot results === #
# Interpolate to find where max_eigenvalues crosses 1
threshold = 1.0
cross_idx = np.where(np.diff(np.sign(np.array(max_eigenvalues) - threshold)))[0]

if len(cross_idx) > 0:
    # Interpolate between the two colonization rates where the crossing happens
    i = cross_idx[0]
    x0, x1 = colonization_rates[i], colonization_rates[i + 1]
    y0, y1 = max_eigenvalues[i], max_eigenvalues[i + 1]

    # Linear interpolation to estimate where the curve crosses y=1
    crossing_c = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)

    # Plot with vertical line
    plt.figure(figsize=(10, 6))
    plt.plot(colonization_rates, max_eigenvalues, marker='o', label='Max Eigenvalue of Product Matrix')
    plt.plot(colonization_rates, mean_final_probs, marker='x', label='Mean Occupancy at Final Time Step')
    plt.axvline(x=crossing_c, color='red', linestyle='--', label=f'λ=1 threshold at c ≈ {crossing_c:.3f}')
    plt.xlabel('Colonization Rate', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.title('Stability and Persistence vs. Colonization Rate', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No crossing of λ=1 found in the colonization rate range.")

# Plot evolution of probabilities
plt.figure(figsize=(10, 6))
for i in range(P.shape[0]):
    plt.plot(P[i, :], lw=0.8)
plt.xlabel('Time step')
plt.ylabel('Occupancy Probability')
plt.title('Occupancy Dynamics Across Patches')
plt.grid(True)
plt.tight_layout()
plt.show()