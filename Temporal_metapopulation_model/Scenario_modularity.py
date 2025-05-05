import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_matrices(n, c, e_0, z, beta, T, scenario_func):
    np.random.seed(42)
    
    # Initialize probability matrix P
    P = np.zeros((n, T))
    P[:, 0] = np.random.random(n)  # Initial condition

    # Initialize the supra-adjacency matrix
    supra_adjacency_matrix = np.zeros((T-1, n, n))

    for k in range(T-1):
        # Use the scenario function to get S and e
        S, e = scenario_func(n, e_0, z, beta,k)
        
        supra_adjacency_matrix[k] = np.eye(n) - np.diag(e) + c * S
        
        q = np.ones(n)
        for i in range(n):
            for j in range(n):
                q[i] *= (1 - c * S[j, i] * P[j, k])
        
        P[:, k + 1] = 1 - (1 - (1 - e) * P[:, k]) * q
    
    return P, supra_adjacency_matrix

def create_scenario_with_modularity(n, e_0, z, beta, k=1, alpha=0.1, square_size=2):
    # Spatial positions
    np.random.seed(42)
    patch_locations = np.random.rand(n, 2) * square_size
    # Homogeneous areas
    A = np.ones(n)
    A_std = A / np.sum(A)

    # Compute pairwise Euclidean distances
    distances = np.linalg.norm(patch_locations[:, None, :] - patch_locations[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)

    # Extinction rates
    e = e_0 * A_std ** (-z)

    # Base colonization matrix: S_ji = A_j^beta * exp(-alpha * d_ji)
    A_j = A_std[None, :] ** beta
    S = A_j * np.exp(-alpha * distances)
    np.fill_diagonal(S, 0)

    # Create SBM adjacency matrix with modular structure
    num_communities = int(np.sqrt(n))
    sizes = [n // num_communities] * num_communities
    p_in = 0.5
    p_out = 0.5 *(1 - k/1000)  # Increase k → lower inter-community connectivity

    # Create the SBM probability matrix
    p_matrix = np.full((num_communities, num_communities), p_out)
    np.fill_diagonal(p_matrix, p_in)

    # Generate modular graph
    G = nx.stochastic_block_model(sizes, p_matrix, seed=42)
    adjacency_matrix = nx.to_numpy_array(G)
    np.fill_diagonal(adjacency_matrix, 0)

    # Apply modular sparsity mask to dispersal matrix
    S *= adjacency_matrix

    return S, e


def max_eigenvalue_product_matrix(supra_adjacency_matrix):
    n = supra_adjacency_matrix.shape[1]
    T_minus_1 = supra_adjacency_matrix.shape[0]
    
    # Initialize the product matrix as an identity matrix
    product_matrix = np.eye(n)
    
    for k in range(T_minus_1):
        product_matrix = np.dot(supra_adjacency_matrix[k], product_matrix)
    
    # Compute the eigenvalues of the product matrix
    eigenvalues = np.linalg.eig(product_matrix)[0]**(1/T_minus_1)
    # Return the maximum eigenvalue
    return np.max(eigenvalues)

# Parameters
n = 30
c = 1
e_0 = 0.01
z = 1
beta = 1


# Vary the colonization rate from 0.1 to 1
colonization_rates = np.linspace(0.6, 1, 10)

# Different values of T
T_values = [100,500]

# Store results for plotting
results = {}

for T in T_values:
    max_eigenvalues = []
    sum_probabilities = []
    
    for c in colonization_rates:
        P, supra_adjacency_matrix = generate_matrices(n, c, e_0, z, beta, T, create_scenario_with_modularity)
        max_eigenvalue = max_eigenvalue_product_matrix(supra_adjacency_matrix)
        max_eigenvalues.append(max_eigenvalue)
        sum_probabilities.append(np.mean(P[:, -1]))
    
    results[T] = (max_eigenvalues, sum_probabilities)

# Plot the results
plt.figure(figsize=(12, 8))

for T in T_values:
    max_eigenvalues, sum_probabilities = results[T]
    plt.plot(colonization_rates, max_eigenvalues, marker='o', label=f'Max Eigenvalue (T={T})')
    plt.plot(colonization_rates, sum_probabilities, marker='x', linestyle='--', label=f'Sum of Probabilities (T={T})')

    # Add vertical dashed lines where max eigenvalue crosses 1
    for i in range(1, len(max_eigenvalues)):
        if max_eigenvalues[i-1] < 1 and max_eigenvalues[i] >= 1:
            plt.axvline(x=colonization_rates[i], color='gray', linestyle='--')
plt.xlabel('Colonization Rate')
plt.ylabel('Value')
plt.title('Max Eigenvalue of Product Matrix and Sum of Probabilities vs. Colonization Rate')
plt.legend()
plt.grid(True)
plt.show()