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

def create_scenario_with_k(n, e_0, z, beta, k):
    distances = np.random.random((n, n)) * 10
    alpha = 0.3
    A = (np.ones(n) * 3) / (1 + np.sqrt(k))  # Decrease A as a function of k
    
    e = e_0 * A**(-z)
    S = A[:, np.newaxis]**beta * np.exp(-alpha * distances)
    
    G = nx.erdos_renyi_graph(n, 1, seed=42)
    adjacency_matrix = nx.to_numpy_array(G)
    np.fill_diagonal(adjacency_matrix, 0)
    S *= adjacency_matrix
    
    row_sums = S.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    S /= row_sums
    
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
n = 100
c = 1
e_0 = 0.1
z = 1
beta = 1


# Vary the colonization rate from 0.1 to 1
colonization_rates = np.linspace(0.1, 1, 10)

# Different values of T
T_values = [100, 200, 500]

# Store results for plotting
results = {}

for T in T_values:
    max_eigenvalues = []
    sum_probabilities = []
    
    for c in colonization_rates:
        P, supra_adjacency_matrix = generate_matrices(n, c, e_0, z, beta, T, create_scenario_with_k)
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