import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import get_cmap

# Fonction pour créer le scénario
def create_scenario_1(n, e_0, z, beta, f, square_size=5, p=1):
    np.random.seed(42)
    patch_locations = np.random.rand(n, 2) * square_size
    A = np.ones(n)
    A_std = A / np.sum(A)
    distances = np.linalg.norm(patch_locations[:, None, :] - patch_locations[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    e = e_0 * A_std ** (-z)
    A_j = A_std[None, :] ** beta
    S = A_j * f(distances)
    np.fill_diagonal(S, 0)
    G = nx.erdos_renyi_graph(n, p, seed=42)
    adjacency_matrix = nx.to_numpy_array(G)
    np.fill_diagonal(adjacency_matrix, 0)
    S *= adjacency_matrix
    return S, e, patch_locations

# Fonction pour tracer le graphe avec couleur selon distance au centre
def plot_graph_colored_by_distance(S, patch_locations, title):
    # Calcul des distances au centre du carré
    center = np.array([2.5, 2.5])
    distances_to_center = np.linalg.norm(patch_locations - center, axis=1)
    
    # Normalisation pour la colormap
    distances_norm = (distances_to_center - distances_to_center.min()) / (distances_to_center.max() - distances_to_center.min())
    
    # Choix de la colormap personnalisée
    cmap = get_cmap('jet')  # 'jet' va du bleu au rouge
    node_colors = [cmap(1 - d) for d in distances_norm]  # inversé pour rouge = centre
    
    G = nx.from_numpy_array(S)  # Graphe non orienté
    
    plt.figure(figsize=(7, 7))
    nx.draw(
        G,
        pos={i: loc for i, loc in enumerate(patch_locations)},
        node_size=100,
        node_color=node_colors,
        with_labels=False,
        edge_color='gray',
        alpha=0.6
    )
    #plt.title(title)
    plt.axis('equal')
    plt.show()

# Paramètres
n = 100
e_0 = 0.01
z = 1
beta = 1
square_size = 10

# Decay function for below the percolation threshold (r = 1)
def decay_below(d):
    return (d < 1).astype(float)

# Decay function for above the percolation threshold (r = 3)
def decay_above(d):
    return (d < 2).astype(float)

# Generate graph below the percolation threshold
S_below, e_below, loc_below = create_scenario_1(n, e_0, z, beta, decay_below, square_size)
plot_graph_colored_by_distance(S_below, loc_below, "Network below percolation threshold (r = 1)")

# Generate graph above the percolation threshold
S_above, e_above, loc_above = create_scenario_1(n, e_0, z, beta, decay_above, square_size)
plot_graph_colored_by_distance(S_above, loc_above, "Network above percolation threshold (r = 3)")