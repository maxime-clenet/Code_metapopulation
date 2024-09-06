import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_matrices(n, c, e_0, z, beta, T, scenario_func, adjacency_matrices, distances, alpha, A):
    np.random.seed(42)
    
    # Initialize probability matrix P
    P = np.zeros((n, T))
    P[:, 0] = np.random.random(n)  # Initial condition

    # Initialize the supra-adjacency matrix
    supra_adjacency_matrix = np.zeros((T-1, n, n))

    for k in range(T-1):
        # Use the scenario function to get S and e
        S, e = scenario_func(n, e_0, z, beta, k, adjacency_matrices, distances, alpha, A)
        
        supra_adjacency_matrix[k] = np.eye(n) - np.diag(e) + c * S
        
        q = np.ones(n)
        for i in range(n):
            for j in range(n):
                q[i] *= (1 - c * S[j, i] * P[j, k])
        
        P[:, k + 1] = 1 - (1 - (1 - e) * P[:, k]) * q

        P[:, k + 1][P[:, k + 1] < 10**-3] = 0
    
    return P, supra_adjacency_matrix

def create_sbm_with_varied_probabilities(n, p_matrix):
    # Number of communities
    num_communities = len(p_matrix)
    
    # Sizes of each community (assuming equal sizes for simplicity)
    sizes = [n // num_communities] * num_communities
    
    # Create the SBM
    G = nx.stochastic_block_model(sizes, p_matrix, seed=42,directed = True)
    
    # Get the adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)
    
    return adjacency_matrix


def create_seasonal_network(n, e_0, z, beta, k, adjacency_matrices, distances, alpha, A):
    e = e_0 * A**(-z)
    S = A[:, np.newaxis]**beta * np.exp(-alpha * distances)
    
    # Determine the adjacency matrix based on k
    # Define the ranges and corresponding adjacency matrices
    ranges = [(0, 200), (200, 250), (250, 450), (450, 500)]
    
    # Determine the adjacency matrix based on k
    for idx, (start, end) in enumerate(ranges):
        if start <= k % 500 < end:
            adjacency_matrix = adjacency_matrices[idx]
            break
    np.fill_diagonal(adjacency_matrix, 0)
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

def min_eigenvalue_all_matrices(supra_adjacency_matrix):
    T_minus_1 = supra_adjacency_matrix.shape[0]
    
    # Initialize a variable to store the minimum of the maximum eigenvalues
    min_eigenvalue = np.inf
    
    for k in range(T_minus_1):
        # Compute the eigenvalues of the k-th supra-adjacency matrix
        eigenvalues = np.linalg.eig(supra_adjacency_matrix[k])[0]
        # Update the minimum eigenvalue if a smaller one is found
        min_eigenvalue = min(min_eigenvalue, np.max(eigenvalues))
    
    return min_eigenvalue

# Parameters
n = 200
distances = np.random.random((n, n)) * 10
alpha =3
A = np.ones(n)
e_0 = 0.1
z = 1.0
beta = 1.0
p_11 = 0.6
p_22 = 0.7
p_12 = 0.02
p_21 = 0.2

p_matrix_1 = np.array([
    [p_11, 0],
    [0, 0]
])

p_matrix_2 = np.array([
    [0, p_12],
    [0, p_22]
])

p_matrix_3 = np.array([
    [0, 0],
    [0, p_22]
])

p_matrix_4 = np.array([
    [p_11, 0],
    [p_21, 0]
])

adjacency_matrix_1 = create_sbm_with_varied_probabilities(n, p_matrix_1)
adjacency_matrix_2 = create_sbm_with_varied_probabilities(n, p_matrix_2)
adjacency_matrix_3 = create_sbm_with_varied_probabilities(n, p_matrix_3)
adjacency_matrix_4 = create_sbm_with_varied_probabilities(n, p_matrix_4)

# Create a tensor of adjacency matrices
adjacency_matrices = np.stack([adjacency_matrix_1, adjacency_matrix_2, adjacency_matrix_3, adjacency_matrix_4])


# Vary the colonization rate from 0.1 to 1
colonization_rates = np.linspace(0.05, 0.7, 30)
#colonization_rates = np.array([0.25])

T_values = [1000]

# Store results for plotting
results = {}

for T in T_values:
    max_eigenvalues = []
    sum_probabilities = []
    min_eigenvalues_all_matrices = []
    
    for c in colonization_rates:
        P, supra_adjacency_matrix = generate_matrices(n, c, e_0, z, beta, T, create_seasonal_network, adjacency_matrices, distances, alpha, A)
        max_eigenvalue = max_eigenvalue_product_matrix(supra_adjacency_matrix)
        min_eigenvalue_all = min_eigenvalue_all_matrices(supra_adjacency_matrix)
        max_eigenvalues.append(max_eigenvalue)
        min_eigenvalues_all_matrices.append(min_eigenvalue_all)
        sum_probabilities.append(np.mean(P[:, -1]))
    
    results[T] = (max_eigenvalues, sum_probabilities, min_eigenvalues_all_matrices)

# Plot the results
plt.figure(figsize=(12, 8))

for T in T_values:
    max_eigenvalues, sum_probabilities, min_eigenvalues_all_matrices = results[T]
    plt.plot(colonization_rates, max_eigenvalues, marker='o', label=f'Max Eigenvalue (T={T})')
    plt.plot(colonization_rates, sum_probabilities, marker='x', linestyle='--', label=f'Mean of Probabilities (T={T})')
    plt.plot(colonization_rates, min_eigenvalues_all_matrices, marker='s', linestyle='-.', label=f'Min Eigenvalue of All Matrices (T={T})')

    # Add vertical dashed lines where max eigenvalue crosses 1
    for i in range(1, len(max_eigenvalues)):
        if max_eigenvalues[i-1] < 1 and max_eigenvalues[i] >= 1:
            plt.axvline(x=colonization_rates[i], color='gray', linestyle='--')

plt.xlabel('Colonization Rate')
plt.ylabel('Value')
plt.title('Max Eigenvalue of Product Matrix, Mean of Probabilities, and Min Eigenvalue of All Matrices vs. Colonization Rate')
plt.legend()
plt.grid(True)
plt.show()