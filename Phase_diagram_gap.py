"""
Phase_diagram_gap.py

Single panel: gap heatmap λ_product − λ_mean in (r_0, A_half) space, w = 1/T.
A_half ∈ [1, 4].
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# ===========================================================================
#  GEOMETRY & HELPERS  (copied from Phase_diagram_thresholds.py)
# ===========================================================================

def periodic_distance(locations, square_size):
    delta = np.abs(locations[:, None, :] - locations[None, :, :])
    delta = np.minimum(delta, square_size - delta)
    return np.sqrt((delta ** 2).sum(axis=2))

def percolation_radius(n, square_size):
    return np.sqrt(4.512 / (np.pi * n / square_size ** 2))

def _build_geometry(n, square_size, e_0, z, beta, seed):
    np.random.seed(seed)
    locs  = np.random.rand(n, 2) * square_size
    A_std = np.ones(n) / n
    dist  = periodic_distance(locs, square_size)
    np.fill_diagonal(dist, np.inf)
    return dist, e_0 * A_std ** (-z), A_std[None, :] ** beta

def make_centered_scenario(r_0, A_half, w_freq, n, square_size, e_0, z, beta, seed):
    dist, e_vec, A_j = _build_geometry(n, square_size, e_0, z, beta, seed)
    def scenario(k):
        r_k = max(0.0, r_0 + A_half * np.sin(2.0 * np.pi * w_freq * k))
        S   = A_j * (dist < r_k).astype(float)
        np.fill_diagonal(S, 0)
        return S.copy(), e_vec.copy()
    return scenario

def build_supra(scenario_func, n, c, T):
    supra = np.zeros((T - 1, n, n))
    for k in range(T - 1):
        S, e     = scenario_func(k)
        supra[k] = np.eye(n) - np.diag(e) + c * S
    return supra

def lambda_product(supra):
    v, log_s = np.ones(supra.shape[1]) / np.sqrt(supra.shape[1]), 0.0
    for M in supra:
        v   = M @ v
        nrm = np.linalg.norm(v)
        if nrm < 1e-300:
            return 0.0
        log_s += np.log(nrm)
        v    /= nrm
    return float(np.exp(log_s / supra.shape[0]))

def lambda_mean(supra):
    return float(np.max(np.abs(np.linalg.eigvals(np.mean(supra, axis=0)))))

def static_lambda(r, n, L, c, e_0, z, beta, seed):
    dist, e_vec, A_j = _build_geometry(n, L, e_0, z, beta, seed)
    S = A_j * (dist < r).astype(float)
    np.fill_diagonal(S, 0)
    M = np.eye(n) - np.diag(e_vec) + c * S
    return float(np.max(np.abs(np.linalg.eigvals(M))))

def find_r_p(n, L, c, e_0, z, beta, seed):
    f = lambda r: static_lambda(r, n, L, c, e_0, z, beta, seed) - 1.0
    r_lo, r_hi = 0.1, L / 2.0
    if f(r_lo) < 0: return r_lo
    if f(r_hi) > 0: return r_hi
    return brentq(f, r_lo, r_hi, xtol=1e-3)


# ===========================================================================
#  PARAMETERS
# ===========================================================================

n, L            = 100, 10   # square_size=10, seed=42 : cohérent avec Test_RGN.py
e_0, z, beta, c = 0.002, 1.0, 1.0, 1.0
T               = 500
SEED            = 42
w               = 1.0 / T

r_c = percolation_radius(n, L)
print("Computing r_p …")
r_p = find_r_p(n, L, c, e_0, z, beta, SEED)
print(f"r_c = {r_c:.3f},  r_p = {r_p:.3f}\n")


# ===========================================================================
#  GRID
# ===========================================================================

r0_arr = np.linspace(0.3, r_p + 2.5, 35)
Ah_arr = np.linspace(0.5, 6.0,        30)

# Static lambda for each r_0 (= λ_product with w=0, same as Test_RGN.py eigvals_w0)
print("Computing static λ for each r_0 …")
ls_vec = np.array([static_lambda(r_0, n, L, c, e_0, z, beta, SEED) for r_0 in r0_arr])
ls_mat = np.tile(ls_vec, (len(Ah_arr), 1))   # broadcast over A_half rows

# Temporal lambda for each (r_0, A_half) with w = 1/T
lp_mat = np.zeros((len(Ah_arr), len(r0_arr)))

for i, A_h in enumerate(Ah_arr):
    for j, r_0 in enumerate(r0_arr):
        scen         = make_centered_scenario(r_0, A_h, w, n, L, e_0, z, beta, SEED)
        supra        = build_supra(scen, n, c, T)
        lp_mat[i, j] = lambda_product(supra)
    print(f"  A_half row {i+1}/{len(Ah_arr)} done")


# ===========================================================================
#  FIGURE
# Gap = λ_product(w=1/T) − λ_product(w=0)
# Matches Test_RGN.py: eigvals_w1[A] − eigvals_w0[A]
# ===========================================================================

gap  = lp_mat - ls_mat
vext = np.max(np.abs(gap))

fig, ax = plt.subplots(figsize=(7, 5))

im = ax.pcolormesh(r0_arr, Ah_arr, gap,
                   cmap='RdBu_r', vmin=-vext, vmax=vext,
                   shading='nearest')

# Temporal persistence boundary
ax.contour(r0_arr, Ah_arr, lp_mat, levels=[1.0],
           colors='k', linewidths=2.5, linestyles='-')
# Static persistence boundary (vertical line at r_0 = r_p)
ax.contour(r0_arr, Ah_arr, ls_mat, levels=[1.0],
           colors='k', linewidths=1.5, linestyles='--')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r'$\lambda_{product}(w{=}1/T) - \lambda_{product}(w{=}0)$', fontsize=11)

ax.set_xlabel(r'Mean radius $r_0$', fontsize=13)
ax.set_ylabel(r'Oscillation amplitude $A_{half}$', fontsize=13)
ax.set_title(
    r'Gap $\lambda(w{=}1/T) - \lambda(w{=}0)$  [cohérent avec Test\_RGN.py]'
    '\n(— seuil temporel,  -- seuil statique $r_0 = r_p$)',
    fontsize=11
)

plt.tight_layout()
plt.savefig('fig_gap_phase_diagram.pdf', bbox_inches='tight')
plt.show()
