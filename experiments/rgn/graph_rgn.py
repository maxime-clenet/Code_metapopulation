"""Random geometric network below and above the percolation radius, nodes coloured by distance to the centre."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from metapop import make_landscape, percolation_radius, threshold_scenario

n = 100
L = 10
e_0, z, beta = 0.01, 1, 1
seed = 42

land = make_landscape(n, L, seed, periodic=False)
r_c = percolation_radius(n, L)


def plot_graph_coloured_by_distance(S, positions, title):
    d_centre = np.linalg.norm(positions - L / 2, axis=1)
    d_norm = (d_centre - d_centre.min()) / (d_centre.max() - d_centre.min())
    cmap = plt.get_cmap('jet')
    plt.figure(figsize=(7, 7))
    nx.draw(nx.from_numpy_array(S), pos=dict(enumerate(positions)), node_size=100,
            node_color=[cmap(1 - d) for d in d_norm],       # red = centre
            with_labels=False, edge_color='gray', alpha=0.6)
    plt.title(title)
    plt.axis('equal')
    plt.show()


for r in (1.0, 2.0):
    S, _ = threshold_scenario(land, lambda k: r, e_0, z, beta)(0)
    side = 'below' if r < r_c else 'above'
    plot_graph_coloured_by_distance(S, land.positions, f'r = {r} ({side} $r_c$ = {r_c:.2f})')
