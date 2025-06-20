import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_scenario_1(n, e_0, z, beta, f, square_size=20, p=1):
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

def periodic_distance(locations, square_size):
    """
    Calcule la matrice des distances entre points avec conditions périodiques sur un carré de taille square_size.
    """
    delta = np.abs(locations[:, None, :] - locations[None, :, :])
    delta = np.minimum(delta, square_size - delta)  # wrap-around effect
    return np.sqrt((delta ** 2).sum(axis=2))


def create_scenario_periodic(n, e_0, z, beta, f, square_size=20, p=1):
    """
    Génère une matrice de connectivité S et un vecteur de taux d’extinction e avec distances périodiques.

    Arguments :
    - n : nombre de patchs
    - e_0 : taux d'extinction de base
    - z : exposant de scaling sur les aires pour l'extinction
    - beta : exposant de scaling sur les aires pour la connectivité
    - f : fonction de décroissance f(d) appliquée aux distances
    - square_size : taille du domaine spatial (carré)
    - p : probabilité d’arête dans le graphe d’Erdős-Rényi

    Retour :
    - S : matrice de connectivité (n x n)
    - e : vecteur des taux d'extinction (n)
    """
    np.random.seed(40)

    # Positions et aires
    patch_locations = np.random.rand(n, 2) * square_size
    A = np.ones(n)
    A_std = A / np.sum(A)

    # Distances périodiques
    distances = periodic_distance(patch_locations, square_size)
    np.fill_diagonal(distances, np.inf)

    # Taux d’extinction
    e = e_0 * A_std ** (-z)

    # Connectivité avec fonction f sur distances périodiques
    A_j = A_std[None, :] ** beta
    S = A_j * f(distances)
    np.fill_diagonal(S, 0)

    # Sparsification par Erdős-Rényi
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
    return np.max(np.abs(eigenvalues))

# === Parameters === #
n = 100
e_0 = 0.001
z = 1
beta = 1
T = 500
c = 1.0
A_values = np.linspace(0.5, 8, 20)

# Scenario generator factory
def make_scenario_A_w(A, w):
    return lambda n, e_0, z, beta, k: create_scenario_periodic(
        n, e_0, z, beta,
        f=lambda d: (d < 0.5 * A * (1 + np.sin(2 * np.pi * w * k))).astype(float)
    )

# Compute eigenvalues for w = 0
eigvals_w0 = []
for A in A_values:
    scenario_func = make_scenario_A_w(A, w=0)
    supra = generate_supra_adjacency_only(n, c, e_0, z, beta, T, scenario_func)
    eigvals_w0.append(max_eigenvalue_product_matrix(supra))

# Compute eigenvalues for w = 1/T
eigvals_w1 = []
for A in A_values:
    scenario_func = make_scenario_A_w(A, w=1/T)
    supra = generate_supra_adjacency_only(n, c, e_0, z, beta, T, scenario_func)
    eigvals_w1.append(max_eigenvalue_product_matrix(supra))

n = 100
L = 20
rho = n / L**2
r_c = np.sqrt(4.512 / (np.pi * rho))  # seuil de percolation en distance
A_threshold = 2 * r_c                 # seuil de distance × 2 interprété comme amplitude


# Plot the two curves
plt.figure(figsize=(8, 5))
plt.plot(A_values, eigvals_w0, marker='o', label='w = 0')
plt.plot(A_values, eigvals_w1, marker='s', label='w = 1/T')
plt.axvline(x=A_threshold, color='red', linestyle='--', label=f'2× Percolation threshold ≈ {A_threshold:.2f}')

plt.xlabel("Amplitude of seasonal fluctuations A")
plt.ylabel(r"Temporal metapopulation capacity $\lambda_{\hat{M}}$")
#plt.title("Max Eigenvalue vs Amplitude A for w=0 and w=1/T")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()