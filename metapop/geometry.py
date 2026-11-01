"""Spatial layout of habitat patches and graph masks."""

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Landscape:
    """
    positions : (n, 2) patch coordinates (None for abstract landscapes)
    areas     : (n,)   patch areas
    distances : (n, n) pairwise distances, diagonal = inf (no self-colonisation)
    size      : side length of the square domain (None if not spatial)
    """
    positions: np.ndarray
    areas: np.ndarray
    distances: np.ndarray
    size: float = None

    @property
    def n(self):
        return len(self.areas)


def distance_matrix(positions, size=None):
    """Euclidean distances; periodic (torus) boundary conditions if `size` is given."""
    delta = np.abs(positions[:, None, :] - positions[None, :, :])
    if size is not None:
        delta = np.minimum(delta, size - delta)
    d = np.sqrt((delta ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    return d


def lattice_positions(n, size):
    """Regular square lattice (n must be a perfect square)."""
    side = int(np.round(np.sqrt(n)))
    xs = np.linspace(0, size, side, endpoint=False)
    return np.array([(x, y) for y in xs for x in xs])[:n]


def make_landscape(n, size, seed=42, random_areas=False, periodic=True, lattice=False):
    """
    Random (or lattice) landscape in a [0, size]² square.

    Areas are U(1, 4) if `random_areas`, else uniform; they are normalised to sum to 1.
    Uses a local RandomState: same draws as the legacy `np.random.seed(seed); np.random.rand(...)`.
    """
    rng = np.random.RandomState(seed)
    positions = lattice_positions(n, size) if lattice else rng.rand(n, 2) * size
    areas = rng.uniform(1, 4, n) if random_areas else np.ones(n)
    areas = areas / areas.sum()
    return Landscape(positions, areas, distance_matrix(positions, size if periodic else None), size)


def percolation_radius(n, size):
    """Critical radius r_c of 2D continuum percolation: n·π·r_c² / L² ≈ 4.512."""
    return np.sqrt(4.512 / (np.pi * n / size ** 2))


def erdos_renyi_mask(n, p, seed=42):
    """Symmetric 0/1 mask from an Erdős–Rényi graph (p = 1 → complete graph)."""
    mask = nx.to_numpy_array(nx.erdos_renyi_graph(n, p, seed=seed))
    np.fill_diagonal(mask, 0)
    return mask


def sbm_mask(sizes, p_matrix, seed=42, directed=False):
    """
    0/1 mask from a stochastic block model.
    For directed graphs, mask[j, i] = 1 means an edge j → i (j can colonise i).
    """
    mask = nx.to_numpy_array(nx.stochastic_block_model(sizes, p_matrix, seed=seed, directed=directed))
    np.fill_diagonal(mask, 0)
    return mask
