import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Number of patches
n = 10

# Generate random areas for each patch
A = np.random.random(n) + 1

# Generate random distances between patches
distances = np.random.random((n, n)) * 10

c = 0.1

# Baseline extinction rate
e_0 = 0.1

# Extinction exponent
z = 1  # You can adjust this value as needed

# Extinction rate for each patch, inversely proportional to area with exponent
e = e_0 * A**(-z)

# Immigration exponent
beta = 1  # You can adjust this value as needed

# Connectivity matrix (dispersal success between patches)
alpha = 0.1  # Example value for alpha
S = A[:, np.newaxis]**beta * np.exp(-alpha * distances)

# Generate an Erdős-Rényi graph
p = 0.1  # Probability of edge creation
G = nx.erdos_renyi_graph(n, p, seed=42)

# Convert the graph to an adjacency matrix
adjacency_matrix = nx.to_numpy_array(G)

# Ensure the diagonal is 0 (each patch is not connected to itself)
np.fill_diagonal(adjacency_matrix, 0)

# Incorporate the adjacency matrix into the connectivity matrix S
S = S * adjacency_matrix

# # Normalize S so that each row sums to 1, avoiding division by zero
# row_sums = S.sum(axis=1, keepdims=True)
# row_sums[row_sums == 0] = 1  # Prevent division by zero
# S = S / row_sums

print(S)
# Time steps
T = 1000

# Initialize occupancy probabilities
P = np.zeros((n, T))
P[:, 0] = np.random.random(n)

# Iterate over time steps
for t in range(T - 1):
    for i in range(n):
        colonization = c * np.sum(P[:, t] * S[:, i]) * (1 - P[i, t])
        extinction = e[i] * P[i, t]
        P[i, t + 1] = P[i, t] + colonization - extinction

print(np.max(np.linalg.eig(c*S-np.diag(e))[0]))

# Plot occupancy probabilities for all patches over time
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(P[i, :], label=f'Patch {i+1}')

plt.xlabel('Time step')
plt.ylabel('Occupancy Probability')
plt.title('Occupancy Probability Evolution for All Patches')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=2)
plt.show()

