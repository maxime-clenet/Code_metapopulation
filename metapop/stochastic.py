"""Continuous-time stochastic (Gillespie) Levins model on a fixed landscape."""

import numpy as np


def gillespie(X0, kernel, areas, c, e, max_time, max_events=np.inf, rng=None):
    """
    Patch states X_i ∈ {0, 1}. Event rates:
      colonisation of empty i : c · A_i · Σ_j X_j kernel[j, i]
      extinction of occupied i: e · X_i / A_i
    Returns (times, states) with states of shape (n_events + 1, n).
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(X0)
    X = X0.copy()
    t, times, states = 0.0, [0.0], [X.copy()]
    while t < max_time and len(states) <= max_events:
        rates = np.concatenate((c * (X @ kernel) * (1 - X) * areas, e * X / areas))
        total = rates.sum()
        if total == 0:
            break
        t += rng.exponential(1 / total)
        i = rng.choice(2 * n, p=rates / total)
        X[i % n] = 1 if i < n else 0
        times.append(t)
        states.append(X.copy())
    return np.array(times), np.array(states)
