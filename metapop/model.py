"""
Discrete-time metapopulation model.

Convention — S[j, i] is the colonisation pressure of source patch j on target patch i:
    S[j, i] = A_j^β · K(d_ji) · mask[j, i]

Nonlinear update:
    q_i(k)   = ∏_j (1 − c·S_ji(k)·P_j(k))
    P_i(k+1) = 1 − (1 − (1 − e_i)·P_i(k)) · q_i(k)

Linearisation around P = 0 (Jacobian, ∂P_i(k+1)/∂P_j(k)):
    M_k = I − diag(e) + c·S_kᵀ
"""

import numpy as np


def extinction_rates(areas, e_0, z):
    """e_i = e_0 · A_i^(−z)."""
    return e_0 * areas ** (-z)


def connectivity(areas, kernel, beta, mask=None):
    """S[j, i] = A_j^β · kernel[j, i] · mask[j, i], zero diagonal."""
    S = areas[:, None] ** beta * kernel
    if mask is not None:
        S = S * mask
    np.fill_diagonal(S, 0)
    return S


def transition_matrix(S, e, c):
    """Jacobian of the nonlinear update at P = 0: M = I − diag(e) + c·Sᵀ."""
    return np.eye(len(e)) - np.diag(e) + c * S.T


def step(P, S, e, c):
    """One step of the nonlinear update."""
    q = np.prod(1 - c * S * P[:, None], axis=0)
    return np.clip(1 - (1 - (1 - e) * P) * q, 0, 1)


def simulate(scenario, c, T, P0, floor=0.0):
    """
    Iterate the nonlinear model for T steps; returns P of shape (n, T + 1).
    `scenario(k)` returns (S_k, e). Occupancies below `floor` are set to 0.
    """
    P = np.zeros((len(P0), T + 1))
    P[:, 0] = P0
    for k in range(T):
        S, e = scenario(k)
        P[:, k + 1] = step(P[:, k], S, e, c)
        P[P[:, k + 1] < floor, k + 1] = 0.0
    return P


def build_supra(scenario, c, T):
    """Array of shape (T, n, n) holding the transition matrices M_0, …, M_{T−1}."""
    return np.stack([transition_matrix(*scenario(k), c) for k in range(T)])


def metapopulation_capacity(landscape, alpha, mask=None):
    """Hanski–Ovaskainen capacity λ_M = ρ(M), M_ij = A_i A_j exp(−α d_ij), M_ii = 0."""
    A = landscape.areas
    M = A[:, None] * A[None, :] * np.exp(-alpha * landscape.distances)
    if mask is not None:
        M = M * mask
    return float(np.max(np.abs(np.linalg.eigvals(M))))
