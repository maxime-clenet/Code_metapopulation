"""Spectral persistence indicators."""

import warnings

import numpy as np
from scipy.optimize import brentq


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def lambda_product(supra):
    """
    Temporal metapopulation capacity λ = ρ(M_{T−1} ··· M_0)^(1/T).
    λ > 1 → persistence, λ < 1 → extinction.

    The running product is renormalised at each step (scales accumulated in log),
    so the result is exact and free of overflow/underflow for any T.
    """
    product = np.eye(supra.shape[1])
    log_scale = 0.0
    for M in supra:
        product = M @ product
        s = np.abs(product).max()
        if s == 0:
            return 0.0
        product /= s
        log_scale += np.log(s)
    rho = spectral_radius(product)
    if rho == 0:
        return 0.0
    return float(np.exp((log_scale + np.log(rho)) / len(supra)))


def lambda_mean(supra):
    """Mean-landscape predictor ρ(M̄), M̄ = (1/T) Σ M_k. Ignores temporal ordering."""
    return spectral_radius(supra.mean(axis=0))


def lambda_shuffled(supra, n_shuffles=20, seed=0):
    """
    Mean and std of λ_product over random permutations of the matrix sequence.

      λ_product − λ_shuffled : effect of temporal autocorrelation (ordering)
      λ_shuffled − λ_mean    : effect of the matrix distribution

    For nonnegative i.i.d. matrices, Jensen gives λ_shuffled ≤ λ_mean asymptotically
    (strict in general): λ_shuffled does not converge to λ_mean.
    """
    rng = np.random.default_rng(seed)
    vals = [lambda_product(supra[rng.permutation(len(supra))]) for _ in range(n_shuffles)]
    return float(np.mean(vals)), float(np.std(vals))


def min_instantaneous_lambda(supra):
    """min_k ρ(M_k): the worst single time step."""
    return min(spectral_radius(M) for M in supra)


def commutator_norm(supra, subsample=50):
    """Mean Frobenius norm of [M_k, M_{k+1}] (= c²·‖[S_kᵀ, S_{k+1}ᵀ]‖) over a subsample of k."""
    step = max(1, len(supra) // subsample)
    return float(np.mean([
        np.linalg.norm(supra[k] @ supra[k + 1] - supra[k + 1] @ supra[k], 'fro')
        for k in range(0, len(supra) - 1, step)
    ]))


def threshold_crossing(x, y, level=1.0):
    """x at which y first crosses `level` (linear interpolation), or None."""
    y = np.asarray(y)
    idx = np.where(np.diff(np.sign(y - level)))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    return x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])


def find_threshold(f, lo, hi, xtol=1e-3):
    """
    Root of the increasing function f on [lo, hi].
    Returns lo if f(lo) ≥ 0 and hi if f(hi) ≤ 0 (with a warning).
    """
    f_lo, f_hi = f(lo), f(hi)
    if f_lo >= 0:
        warnings.warn(f"f ≥ 0 already at lower bound {lo}; threshold set to {lo}")
        return lo
    if f_hi <= 0:
        warnings.warn(f"f ≤ 0 still at upper bound {hi}; threshold set to {hi}")
        return hi
    return brentq(f, lo, hi, xtol=xtol)
