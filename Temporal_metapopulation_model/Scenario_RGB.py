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
        S, e = scenario_func(n, e_0, z, beta, k)
        supra_adjacency_matrix[k] = np.eye(n) - np.diag(e) + c * S

        # Nonlinear update
        q = np.ones(n)
        for i in range(n):
            for j in range(n):
                q[i] *= (1 - c * S[j, i] * P[j, k])
        P[:, k + 1] = 1 - (1 - (1 - e) * P[:, k]) * q
        P[:, k + 1] = np.clip(P[:, k + 1], 0, 1)

    return P, supra_adjacency_matrix

def create_scenario_1(n, e_0, z, beta, f, square_size=5, p=1):
    """
    Génère une matrice de connectivité S et un vecteur de taux d’extinction e, avec connectivité modulée
    par une fonction f appliquée à la distance entre patchs.

    Arguments :
    - n : nombre de patchs
    - e_0 : taux d'extinction de base
    - z : exposant de scaling sur les aires pour l'extinction
    - beta : exposant de scaling sur les aires pour la connectivité
    - f : fonction de décroissance f(d) appliquée aux distances (ex: lambda d: (d < seuil).astype(float))
    - square_size : taille du carré spatial
    - p : probabilité d’arête dans le graphe d’Erdős-Rényi

    Retour :
    - S : matrice de connectivité (n x n)
    - e : vecteur des taux d'extinction (n)
    """

    np.random.seed(42)

    # Génération des positions et aires
    patch_locations = np.random.rand(n, 2) * square_size
    # A = np.random.uniform(1, 4, n)
    A = np.ones(n)
    A_std = A / np.sum(A)

    # Distances spatiales
    distances = np.linalg.norm(patch_locations[:, None, :] - patch_locations[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)

    # Taux d’extinction
    e = e_0 * A_std ** (-z)

    # Connectivité avec fonction f
    A_j = A_std[None, :] ** beta
    S = A_j * f(distances)
    np.fill_diagonal(S, 0)

    # Sparsification avec graphe aléatoire
    G = nx.erdos_renyi_graph(n, p, seed=42)
    adjacency_matrix = nx.to_numpy_array(G)
    np.fill_diagonal(adjacency_matrix, 0)
    S *= adjacency_matrix

    return S, e

def generate_supra_adjacency_only(n, c, e_0, z, beta, T, scenario_func):
    """
    Génère uniquement la matrice supra-adjacente temporelle pour un modèle spatio-temporel.

    Paramètres :
    - n : nombre de patchs
    - c : taux de colonisation
    - e_0, z : paramètres d’extinction
    - beta : pondération de l’aire sur la connectivité
    - T : durée temporelle
    - scenario_func : fonction de génération S, e avec dépendance temporelle (doit accepter k)

    Retour :
    - supra_adjacency_matrix : tableau de forme (T-1, n, n)
    """
    np.random.seed(42)
    supra_adjacency_matrix = np.zeros((T - 1, n, n))

    for k in range(T - 1):
        S, e = scenario_func(n, e_0, z, beta, k)
        supra_adjacency_matrix[k] = np.eye(n) - np.diag(e) + c * S

    return supra_adjacency_matrix

def max_eigenvalue_product_matrix(supra_adjacency_matrix):
    n = supra_adjacency_matrix.shape[1]
    T_minus_1 = supra_adjacency_matrix.shape[0]

    product_matrix = np.eye(n)
    for k in range(T_minus_1):
        product_matrix = np.dot(supra_adjacency_matrix[k], product_matrix)

    eigenvalues = np.linalg.eigvals(product_matrix)**(1/T_minus_1)
    return np.max(eigenvalues.real)

# === Parameters === #
n = 50
e_0 = 0.01
z = 1
beta = 1
T = 500
colonization_rates = np.linspace(0.1, 2, 20)

# === Define time-varying decay function === #
def time_varying_threshold_decay(d, k, A=4.0, w=0.01):
    r = 0.5 * A * (1 + np.sin(2 * np.pi * w * k))
    # r = 2
    return (d < r).astype(float)

# === Wrapper for create_scenario_1 with f === #
def scenario_with_threshold(n, e_0, z, beta, k):
    return create_scenario_1(
        n, e_0, z, beta,
        f=lambda d: time_varying_threshold_decay(d, k)
    )

# === Run simulation over varying colonization rates === #
max_eigenvalues = []
mean_final_probs = []

for c in colonization_rates:
    supra = generate_supra_adjacency_only(n, c, e_0, z, beta, T, scenario_with_threshold)
    max_eigenvalue = max_eigenvalue_product_matrix(supra)
    max_eigenvalues.append(max_eigenvalue)
    # mean_final_probs.append(np.mean(P[:, -1]))

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
    # plt.plot(colonization_rates, mean_final_probs, marker='x', label='Mean Occupancy at Final Time Step')
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


# === Experiment 1: max eigenvalue vs colonization rate c === #
colonization_rates = np.linspace(0.1, 2, 20)
def scenario_c(n, e_0, z, beta, k, A=4.0, w=0):
    return create_scenario_1(n, e_0, z, beta, f=lambda d: (d < 0.5 * A * (1 + np.sin(2 * np.pi * w * k))).astype(float))

eigvals_c = []
for c in colonization_rates:
    supra = generate_supra_adjacency_only(n, c, e_0, z, beta, T, scenario_c)
    eigvals_c.append(max_eigenvalue_product_matrix(supra))

# === Experiment 2: max eigenvalue vs amplitude A === #
A_values = np.linspace(0, 0.5, 20)
def make_scenario_A(A):
    return lambda n, e_0, z, beta, k: create_scenario_1(
        n, e_0, z, beta,
        f=lambda d: (d < 0.5 * A * (1 + np.sin(2 * np.pi * 1/T * k))).astype(float)
    )

eigvals_A = []
for A in A_values:
    supra = generate_supra_adjacency_only(n, 1.0, e_0, z, beta, T, make_scenario_A(A))
    eigvals_A.append(max_eigenvalue_product_matrix(supra))

# === Experiment 3: max eigenvalue vs frequency w === #
w_values = np.array([i / T for i in range(0, 11)])
def make_scenario_w(w):
    return lambda n, e_0, z, beta, k: create_scenario_1(
        n, e_0, z, beta,
        f=lambda d: (d < 0.5 * 2 * (1 + np.sin(2 * np.pi * w * k))).astype(float)
    )

eigvals_w = []
for w in w_values:
    supra = generate_supra_adjacency_only(n, 2.0, e_0, z, beta, T, make_scenario_w(w))
    eigvals_w.append(max_eigenvalue_product_matrix(supra))

# === Plotting all three graphs === #
fig, axs = plt.subplots(3, 1, figsize=(10, 16))

# Plot 1: eigenvalue vs c
axs[0].plot(colonization_rates, eigvals_c, marker='o')
axs[0].set_title("Max Eigenvalue vs Colonization Rate (c)", fontsize=14)
axs[0].set_xlabel("Colonization Rate (c)")
axs[0].set_ylabel("Max Eigenvalue")
axs[0].grid(True)

# Plot 2: eigenvalue vs A
axs[1].plot(A_values, eigvals_A, marker='s', color='green')
axs[1].set_title("Max Eigenvalue vs Amplitude (A)", fontsize=14)
axs[1].set_xlabel("Amplitude A of Threshold Oscillation")
axs[1].set_ylabel("Max Eigenvalue")
axs[1].grid(True)

# Plot 3: eigenvalue vs w
axs[2].plot(w_values, eigvals_w, marker='^', color='purple')
axs[2].set_title("Max Eigenvalue vs Frequency (w)", fontsize=14)
axs[2].set_xlabel("Frequency w of Threshold Oscillation")
axs[2].set_ylabel("Max Eigenvalue")
axs[2].grid(True)

plt.tight_layout()
plt.show()