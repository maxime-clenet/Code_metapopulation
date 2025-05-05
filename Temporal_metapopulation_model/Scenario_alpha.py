import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Existing scenario function (with alpha decreasing over time)
def create_scenario_alpha_decay(n, e_0, z, beta, t, T, alpha_0=0.1, square_size=2, p=1):
    alpha_t = alpha_0 * (1 + t / 2000)
    patch_locations = np.random.rand(n, 2) * square_size
    A = np.random.uniform(1, 4, n)
    A_std = A / np.sum(A)
    distances = np.linalg.norm(patch_locations[:, None, :] - patch_locations[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    e = e_0 * A_std ** (-z)
    A_j = A_std[None, :] ** beta
    S = A_j * np.exp(-alpha_t * distances)
    np.fill_diagonal(S, 0)
    G = nx.erdos_renyi_graph(n, p, seed=42)
    adjacency_matrix = nx.to_numpy_array(G)
    np.fill_diagonal(adjacency_matrix, 0)
    S *= adjacency_matrix
    return S, e

# Updated generator
def generate_matrices(n, c, e_0, z, beta, T, scenario_func):
    np.random.seed(42)
    P = np.zeros((n, T))
    P[:, 0] = np.random.random(n)
    supra_adjacency_matrix = np.zeros((T-1, n, n))
    for k in range(T-1):
        S, e = scenario_func(n, e_0, z, beta, k, T)
        supra_adjacency_matrix[k] = np.eye(n) - np.diag(e) + c * S
        q = np.ones(n)
        for i in range(n):
            for j in range(n):
                q[i] *= (1 - c * S[j, i] * P[j, k])
        P[:, k + 1] = 1 - (1 - (1 - e) * P[:, k]) * q
        P[:, k + 1] = np.clip(P[:, k + 1], 0, 1)
    return P, supra_adjacency_matrix

# Eigenvalue extraction
def max_eigenvalue_product_matrix(supra_adjacency_matrix):
    product_matrix = np.eye(supra_adjacency_matrix.shape[1])
    for M in supra_adjacency_matrix:
        product_matrix = M @ product_matrix
    eigenvalues = np.linalg.eigvals(product_matrix)**(1 / supra_adjacency_matrix.shape[0])
    return np.max(eigenvalues.real)

# Parameters
n = 10
e_0 = 0.01
z = 1
beta = 1
colonization_rates = np.linspace(0.1, 0.2, 10)
T_values = [50, 200, 2000]

# Plot results for each T
plt.figure(figsize=(10, 6))
for T in T_values:
    max_eigenvalues = []
    for c in colonization_rates:
        P, supra = generate_matrices(n, c, e_0, z, beta, T, create_scenario_alpha_decay)
        max_eig = max_eigenvalue_product_matrix(supra)
        max_eigenvalues.append(max_eig)
    plt.plot(colonization_rates, max_eigenvalues, marker='o', label=f'T = {T}')

plt.axhline(y=1, color='grey', linestyle='--', label='Threshold λ = 1')
plt.xlabel('Colonization Rate', fontsize=14)
plt.ylabel('Max Eigenvalue of Product Matrix', fontsize=14)
plt.title('Effect of T on Stability Threshold (λ)', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()