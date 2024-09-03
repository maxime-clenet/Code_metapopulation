import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Parameters
n = 10  # Number of patches
c = 0.1  # Fixed colonization rate for all patches
e_0 = 0.1  # Baseline extinction rate
z = 1  # Extinction exponent
beta = 1  # Emmigration exponent
alpha = 0.1  # Example value for alpha
p = 0.2  # Probability of edge creation in Erdős-Rényi graph
T = 1000  # Time steps

# Generate random areas for each patch
A = np.random.random(n) + 1

# Generate random distances between patches
distances = np.random.random((n, n)) * 10

# Calculate extinction rate for each patch
e = e_0 * A**(-z)

# Calculate the connectivity matrix (dispersal success between patches)
S = A[:, np.newaxis]**beta * np.exp(-alpha * distances)

# Generate an Erdős-Rényi graph
G = nx.erdos_renyi_graph(n, p, seed=42)

# Convert the graph to an adjacency matrix and ensure diagonal is 0
adjacency_matrix = nx.to_numpy_array(G)
np.fill_diagonal(adjacency_matrix, 0)

# Incorporate the adjacency matrix into the connectivity matrix S
S *= adjacency_matrix

# # Normalize S so that each row sums to 1, avoiding division by zero
# row_sums = S.sum(axis=1, keepdims=True)
# row_sums[row_sums == 0] = 1  # Prevent division by zero
# S /= row_sums

# Initialize a matrix P with zeros, of size n x T
P = np.zeros((n, T))

# Set the initial condition: random probabilities for each patch
P[:, 0] = np.random.random(n)

# Iterate over time steps
for k in range(T-1):
    # Initialize q as an array of ones for the product calculation
    q = np.ones(n)
    # Calculate the product term for all nodes using nested loops
    for i in range(n):
        for j in range(n):
            q[i] *= (1 - c * S[j, i] * P[j, k])
    # Update the probabilities for all nodes at time k+1
    P[:, k + 1] = 1 - (1 - (1 - e) * P[:, k]) * q

# Print the maximum eigenvalue of the connectivity matrix S
print("Max eigenvalue of the connectivity matrix S:", np.max(np.linalg.eig(c*S - np.diag(e))[0]))

# Plot the probability evolution for all patches
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(P[i, :], label=f'Patch {i+1}')
plt.xlabel('Time step')
plt.ylabel('Probability')
plt.title('Probability Evolution for All Patches')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=2)
plt.show()
