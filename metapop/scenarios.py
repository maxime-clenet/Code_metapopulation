"""
Time-varying landscapes.

A scenario is a callable `scenario(k) -> (S_k, e)` (see `metapop.model` for conventions).
"""

import numpy as np

from .geometry import sbm_mask
from .model import connectivity, extinction_rates, transition_matrix
from .spectral import find_threshold, spectral_radius


# ---------------------------------------------------------------------------
#  Generic scenario
# ---------------------------------------------------------------------------

def make_scenario(landscape, e_0, z, beta, kernel, mask=None):
    """
    kernel(d, k) -> (n, n) dispersal kernel at time k.
    mask: None, a fixed (n, n) 0/1 array, or a callable k -> array.
    """
    e = extinction_rates(landscape.areas, e_0, z)

    def scenario(k):
        m = mask(k) if callable(mask) else mask
        return connectivity(landscape.areas, kernel(landscape.distances, k), beta, m), e

    return scenario


def threshold_kernel(radius):
    """Hard-threshold kernel: K(d) = 1 if d < r(k), with r = radius(k)."""
    return lambda d, k: (d < radius(k)).astype(float)


def exponential_kernel(alpha):
    """K(d) = exp(−α(k)·d), with α = alpha(k)."""
    return lambda d, k: np.exp(-alpha(k) * d)


def threshold_scenario(landscape, radius, e_0, z, beta, mask=None):
    return make_scenario(landscape, e_0, z, beta, threshold_kernel(radius), mask)


# ---------------------------------------------------------------------------
#  Connectivity radius signals r(k)
# ---------------------------------------------------------------------------

def sinusoidal_radius(A, w):
    """r(k) = (A/2)·(1 + sin(2π·w·k)), oscillates in [0, A]."""
    return lambda k: 0.5 * A * (1.0 + np.sin(2.0 * np.pi * w * k))


def centered_radius(r_0, A_half, w):
    """r(k) = max(0, r_0 + A_half·sin(2π·w·k))."""
    return lambda k: max(0.0, r_0 + A_half * np.sin(2.0 * np.pi * w * k))


def square_wave_radius(A_on, phi_on, w):
    """r(k) = A_on during a fraction phi_on of each period, 0 otherwise (mean phi_on·A_on)."""
    return lambda k: A_on if (w * k) % 1.0 < phi_on else 0.0


def duty_cycle(A, r_c):
    """
    Fraction of time the sinusoidal r(t) = (A/2)(1 + sin) exceeds r_c (independent of w):
    φ(A) = ½ − arcsin(2·r_c/A − 1)/π.
    """
    if A <= r_c:
        return 0.0
    return 0.5 - np.arcsin(np.clip(2.0 * r_c / A - 1.0, -1.0, 1.0)) / np.pi


def amplitude_for_duty_cycle(phi, r_c):
    """Inverse of `duty_cycle`: A = 2·r_c / (1 + cos(π·φ))."""
    if phi <= 0:
        return r_c
    if phi >= 1.0:
        return np.inf
    return 2.0 * r_c / (1.0 + np.cos(np.pi * phi))


# ---------------------------------------------------------------------------
#  Static persistence threshold
# ---------------------------------------------------------------------------

def static_lambda(landscape, r, c, e_0, z, beta):
    """ρ(M(r)) for the static threshold network of radius r."""
    S, e = threshold_scenario(landscape, lambda k: r, e_0, z, beta)(0)
    return spectral_radius(transition_matrix(S, e, c))


def persistence_radius(landscape, c, e_0, z, beta, r_lo=0.1, r_hi=None):
    """r_p such that ρ(M(r_p)) = 1 (static persistence threshold)."""
    r_hi = landscape.size / 2.0 if r_hi is None else r_hi
    return find_threshold(lambda r: static_lambda(landscape, r, c, e_0, z, beta) - 1.0, r_lo, r_hi)


# ---------------------------------------------------------------------------
#  Seasonal (piecewise-constant) schedules
# ---------------------------------------------------------------------------

def cyclic_schedule(items, durations):
    """k -> items[i], where phase i lasts durations[i] steps and the cycle repeats."""
    bounds = np.cumsum(durations)
    return lambda k: items[int(np.searchsorted(bounds, k % bounds[-1], side='right'))]


def two_group_seasonal_masks(n, p_11, p_12, p_21, p_22, seed=42):
    """
    Four directed SBM masks for two equal groups of patches (group 1 = first half):
      phase 1: within group 1        phase 2: 1 → 2 and within group 2
      phase 3: within group 2        phase 4: 2 → 1 and within group 1
    """
    p_matrices = [
        [[p_11, 0], [0, 0]],
        [[0, p_12], [0, p_22]],
        [[0, 0], [0, p_22]],
        [[p_11, 0], [p_21, 0]],
    ]
    return [sbm_mask([n // 2, n // 2], p, seed=seed, directed=True) for p in p_matrices]
