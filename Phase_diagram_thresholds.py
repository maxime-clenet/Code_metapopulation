"""
Phase_diagram_thresholds.py

Phase diagram in (r_0, A_half) parameter space for w = 1/T.

Two thresholds to distinguish:
  r_c : geometric percolation of the RGG (giant component appears)
        n·π·r_c² / L² ≈ 4.51  on the torus
  r_p : metapopulation persistence threshold, ρ(M(r_p)) = 1
        in general r_p > r_c (having a giant component is necessary but not sufficient)

Scenario: r(k) = max(0,  r_0 + A_half · sin(2π · w · k))
  r_0    = mean connectivity radius  (oscillation centre)
  A_half = oscillation amplitude

Three amplitude regimes for fixed r_0 < r_c < r_p:
  (1)  A_half < r_c  − r_0  : r(t) always below r_c  → fully fragmented, no rescue
  (2)  r_c − r_0 < A_half < r_p − r_0 : r(t) crosses r_c but not r_p → giant
       component appears transiently but K^(t) never expansive → no rescue yet
  (3)  A_half > r_p − r_0  : r(t) crosses r_p → periodic rescue operates;
       persistence possible even when ρ(M(r_0)) < 1

Key prediction: λ_product is non-monotone in A_half for fixed r_0 < r_p.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import brentq


# ===========================================================================
#  GEOMETRY
# ===========================================================================

def periodic_distance(locations, square_size):
    delta = np.abs(locations[:, None, :] - locations[None, :, :])
    delta = np.minimum(delta, square_size - delta)
    return np.sqrt((delta ** 2).sum(axis=2))


def percolation_radius(n, square_size):
    """r_c from n·π·r_c²/L² = 4.512."""
    return np.sqrt(4.512 / (np.pi * n / square_size ** 2))


def _build_geometry(n, square_size, e_0, z, beta, seed):
    np.random.seed(seed)
    locs  = np.random.rand(n, 2) * square_size
    A_std = np.ones(n) / n
    dist  = periodic_distance(locs, square_size)
    np.fill_diagonal(dist, np.inf)
    return dist, e_0 * A_std ** (-z), A_std[None, :] ** beta


# ===========================================================================
#  SCENARIO: centred sinusoidal
# ===========================================================================

def make_centered_scenario(r_0, A_half, w_freq, n, square_size,
                            e_0, z, beta, seed):
    """
    r(k) = max(0,  r_0 + A_half · sin(2π·w·k)).
    For A_half = 0: static network.
    For A_half > r_p − r_0: r(t) crosses the persistence threshold r_p.
    For A_half > r_0:       r(t) clips at 0 (complete disconnection phase).
    """
    dist, e_vec, A_j = _build_geometry(n, square_size, e_0, z, beta, seed)

    def scenario(k):
        r_k = max(0.0, r_0 + A_half * np.sin(2.0 * np.pi * w_freq * k))
        S   = A_j * (dist < r_k).astype(float)
        np.fill_diagonal(S, 0)
        return S.copy(), e_vec.copy()

    return scenario


# ===========================================================================
#  SUPRA-ADJACENCY AND SPECTRAL METRICS
# ===========================================================================

def build_supra(scenario_func, n, c, T):
    supra = np.zeros((T - 1, n, n))
    for k in range(T - 1):
        S, e     = scenario_func(k)
        supra[k] = np.eye(n) - np.diag(e) + c * S
    return supra


def lambda_product(supra):
    """Temporal metapopulation capacity via power iteration (overflow-safe)."""
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
    """Mean-matrix spectral radius."""
    return float(np.max(np.abs(np.linalg.eigvals(np.mean(supra, axis=0)))))


# ===========================================================================
#  STATIC LAMBDA AND THRESHOLD r_p
# ===========================================================================

def static_lambda(r, n, L, c, e_0, z, beta, seed):
    """ρ(M(r)) for a static network at connectivity radius r."""
    dist, e_vec, A_j = _build_geometry(n, L, e_0, z, beta, seed)
    S = A_j * (dist < r).astype(float)
    np.fill_diagonal(S, 0)
    M = np.eye(n) - np.diag(e_vec) + c * S
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def find_r_p(n, L, c, e_0, z, beta, seed, r_lo=0.1):
    """Binary search: ρ(M(r_p)) = 1."""
    r_hi = L / 2.0
    f    = lambda r: static_lambda(r, n, L, c, e_0, z, beta, seed) - 1.0
    if f(r_lo) < 0:
        print(f"  [warn] static λ < 1 even at r={r_lo:.2f}; r_p estimated as {r_lo}")
        return r_lo
    if f(r_hi) > 0:
        print(f"  [warn] static λ > 1 even at r={r_hi:.2f}; r_p estimated as {r_hi}")
        return r_hi
    return brentq(f, r_lo, r_hi, xtol=1e-3)


# ===========================================================================
#  PARAMETERS
# ===========================================================================

n, L       = 100, 20
e_0, z, beta, c = 0.001, 1.0, 1.0, 1.0
T          = 500
SEED       = 40
w          = 1.0 / T          # one full cycle over the simulation window

r_c = percolation_radius(n, L)
print("Computing r_p (binary search on static λ) …")
r_p = find_r_p(n, L, c, e_0, z, beta, SEED)
print(f"r_c (geometric percolation)  = {r_c:.3f}")
print(f"r_p (metapop persistence)    = {r_p:.3f}")
print(f"r_p / r_c                    = {r_p/r_c:.2f}\n")


# ===========================================================================
#  FIG 0 — Static λ(r): verify r_c and r_p
# ===========================================================================

print("Fig 0: static λ scan …")
r_scan  = np.linspace(0.3, r_p + 2.5, 50)
lam_s   = [static_lambda(r, n, L, c, e_0, z, beta, SEED) for r in r_scan]

fig0, ax = plt.subplots(figsize=(8, 4))
ax.plot(r_scan, lam_s, '-o', ms=4, color='steelblue', lw=2)
ax.axhline(1, color='k', ls=':', lw=1.2, label='Threshold $\\lambda = 1$')
ax.axvline(r_c, color='royalblue', ls='--', lw=2,
           label=f'$r_c$ = {r_c:.2f}  (geometric percolation)')
ax.axvline(r_p, color='crimson', ls='--', lw=2,
           label=f'$r_p$ = {r_p:.2f}  (persistence threshold)')
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 2 else 2],
                 r_c, r_p, color='orange', alpha=0.12,
                 label='$r_c < r < r_p$: giant component but sub-critical')
ax.set_xlabel('Connectivity radius $r$', fontsize=12)
ax.set_ylabel(r'$\rho(M(r))$ — static spectral radius', fontsize=11)
ax.set_title('Static metapopulation capacity vs connectivity radius', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig0_static_lambda.pdf', bbox_inches='tight')
plt.show()
print()


# ===========================================================================
#  FIG 1 — Phase diagram in (r_0, A_half)  [w = 1/T]
#
#  Annotation lines (all independent of w):
#    r_0 = r_c   (vertical, blue)   : static percolation
#    r_0 = r_p   (vertical, red)    : static persistence
#    A_half = r_p − r_0  (diagonal) : r(t) starts crossing r_p
#    A_half = r_c − r_0  (diagonal) : r(t) starts crossing r_c
# ===========================================================================

print("Fig 1: phase diagram (r_0, A_half) …")
r0_arr = np.linspace(0.3,  r_p + 2.5, 30)
Ah_arr = np.linspace(0.0,  r_p + 1.5, 28)

lp_ph  = np.zeros((len(Ah_arr), len(r0_arr)))
lm_ph  = np.zeros_like(lp_ph)

for i, A_h in enumerate(Ah_arr):
    for j, r_0 in enumerate(r0_arr):
        scen      = make_centered_scenario(r_0, A_h, w, n, L, e_0, z, beta, SEED)
        supra     = build_supra(scen, n, c, T)
        lp_ph[i, j] = lambda_product(supra)
        lm_ph[i, j] = lambda_mean(supra)
    print(f"  A_half row {i+1}/{len(Ah_arr)} done")

# --- persistence boundary lines ---
r0_sub_rp = r0_arr[r0_arr <= r_p]
r0_sub_rc = r0_arr[r0_arr <= r_c]

def annotate_thresholds(ax):
    ax.axvline(r_c, color='royalblue', ls='--', lw=2,
               label=f'$r_c$ = {r_c:.2f}')
    ax.axvline(r_p, color='crimson',   ls='--', lw=2,
               label=f'$r_p$ = {r_p:.2f}')
    ax.plot(r0_sub_rp, r_p - r0_sub_rp, color='red', ls='-.',
            lw=1.8, label=r'$A_{half} = r_p - r_0$  (crosses $r_p$)')
    ax.plot(r0_sub_rc, r_c - r0_sub_rc, color='royalblue', ls='-.',
            lw=1.2, label=r'$A_{half} = r_c - r_0$  (crosses $r_c$)')
    ax.set_xlabel('Mean radius $r_0$', fontsize=12)
    ax.set_ylabel('Oscillation amplitude $A_{half}$', fontsize=11)
    ax.legend(fontsize=7.5, loc='upper right')

fig1, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig1.suptitle(
    r'Phase Diagram $(r_0,\; A_{half})$ for $w = 1/T$'
    f'\n[$r_c$ = {r_c:.2f},  $r_p$ = {r_p:.2f},  '
    r'$r(k) = \max(0,\; r_0 + A_{half}\sin(2\pi wk))$]',
    fontsize=12
)

# Panel A: λ_product heatmap
im0 = axes[0].pcolormesh(r0_arr, Ah_arr, lp_ph,
                          cmap='RdYlGn', vmin=0.5, vmax=1.5, shading='nearest')
axes[0].contour(r0_arr, Ah_arr, lp_ph, levels=[1.0], colors='k', linewidths=2.5)
plt.colorbar(im0, ax=axes[0], label=r'$\lambda_{product}$')
annotate_thresholds(axes[0])
axes[0].set_title(r'Temporal metapopulation capacity $\lambda_{product}$'
                  '\n(black contour: persistence boundary)', fontsize=11)

# Panel B: gap λ_product − λ_mean
gap   = lp_ph - lm_ph
vext  = max(abs(gap.min()), abs(gap.max())) + 1e-6
im1   = axes[1].pcolormesh(r0_arr, Ah_arr, gap,
                            cmap='RdBu_r', vmin=-vext, vmax=vext, shading='nearest')
axes[1].contour(r0_arr, Ah_arr, lp_ph, levels=[1.0], colors='k',
                linewidths=2.5, linestyles='-')
axes[1].contour(r0_arr, Ah_arr, lm_ph, levels=[1.0], colors='k',
                linewidths=1.5, linestyles='--')
plt.colorbar(im1, ax=axes[1], label=r'$\lambda_{product} - \lambda_{mean}$')
annotate_thresholds(axes[1])
axes[1].set_title(
    r'Gap $\lambda_{product} - \lambda_{mean}$'
    '\n(— temporal threshold, -- mean-field threshold)',
    fontsize=11
)

plt.tight_layout()
plt.savefig('fig1_phase_diagram_r0_Ahalf.pdf', bbox_inches='tight')
plt.show()
print()


# ===========================================================================
#  FIG 2 — Non-monotone rescue: λ vs A_half for r_0 < r_p
#
#  Three regimes (for r_0 < r_c < r_p) annotated:
#    I   A_half < r_c − r_0  : always below r_c (fragmented)
#    II  r_c − r_0 < A_half < r_p − r_0 : crosses r_c, still sub-critical
#    III A_half > r_p − r_0  : crosses r_p → periodic rescue
#
#  Non-monotone result: λ_product peaks at some optimal A_half* then
#  decreases as the disconnected phase (r → 0) causes near-total extinction.
# ===========================================================================

print("Fig 2: non-monotone A_half scan …")

# Choose r_0 values at different positions relative to r_c and r_p
r0_choices = np.array([
    max(0.4, r_c - 1.5),   # well below r_c
    max(0.4, r_c - 0.5),   # just below r_c
    (r_c + r_p) / 2,       # between r_c and r_p
    r_p - 0.3,             # just below r_p
])
r0_labels = [
    f'$r_0$ = {r:.2f}  (below $r_c$, regime I–III)'
    if r < r_c else
    f'$r_0$ = {r:.2f}  (between $r_c$ and $r_p$, regime III only)'
    for r in r0_choices
]
COLORS = ['steelblue', 'darkorange', 'seagreen', 'crimson']

Ah_scan = np.linspace(0.0, r_p + 1.8, 40)

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.5))
fig2.suptitle(
    r'Non-Monotone Periodic Rescue: $\lambda_{product}$ vs $A_{half}$  [$w = 1/T$]'
    f'\n(all $r_0 < r_p = {r_p:.2f}$, static sub-critical)',
    fontsize=12
)

for r_0, label, color in zip(r0_choices, r0_labels, COLORS):
    lp_s, lm_s = [], []
    for A_h in Ah_scan:
        scen  = make_centered_scenario(r_0, A_h, w, n, L, e_0, z, beta, SEED)
        supra = build_supra(scen, n, c, T)
        lp_s.append(lambda_product(supra))
        lm_s.append(lambda_mean(supra))
    lp_s, lm_s = np.array(lp_s), np.array(lm_s)

    # Panel A: λ values
    axes2[0].plot(Ah_scan, lp_s, '-o',  color=color, ms=3, lw=2,   label=label)
    axes2[0].plot(Ah_scan, lm_s, '--',  color=color, lw=1.2, alpha=0.6)

    # Panel B: gap
    axes2[1].plot(Ah_scan, lp_s - lm_s, '-o', color=color, ms=3, lw=2, label=label)

    # Mark regime boundaries (only if r_0 < r_p)
    for ax in axes2:
        if r_0 < r_c:
            ax.axvline(r_c - r_0, color=color, ls=':', lw=1.0, alpha=0.5)
        ax.axvline(r_p - r_0, color=color, ls='--', lw=1.2, alpha=0.8)

    print(f"  r_0 = {r_0:.2f} done")

# Shade the three regime regions (using r_0 = r0_choices[0] as reference)
r0_ref = r0_choices[0]
if r0_ref < r_c:
    for ax in axes2:
        ylo, yhi = ax.get_ylim() if ax.get_ylim()[1] > 0.1 else (-0.1, 1.5)
        ax.axvspan(0,             r_c - r0_ref, alpha=0.06, color='blue',
                   label='Regime I (always below $r_c$)')
        ax.axvspan(r_c - r0_ref, r_p - r0_ref, alpha=0.06, color='orange',
                   label='Regime II (crosses $r_c$, not $r_p$)')
        ax.axvspan(r_p - r0_ref, Ah_scan[-1],  alpha=0.06, color='green',
                   label='Regime III (crosses $r_p$ → rescue)')

for ax in axes2:
    ax.axhline(0 if ax == axes2[1] else 1, color='k', ls=':', lw=1.2)
    ax.set_xlabel(r'Oscillation amplitude $A_{half}$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5)

axes2[0].set_ylabel(r'$\lambda$ (— product,  – – mean)', fontsize=11)
axes2[0].set_title(
    r'$\lambda_{product}$ and $\lambda_{mean}$ vs $A_{half}$'
    '\n(vertical dashes: $A_{half} = r_p - r_0$)', fontsize=11
)
axes2[1].set_ylabel(r'Gap $\lambda_{product} - \lambda_{mean}$', fontsize=11)
axes2[1].set_title(
    'Gap: temporal product minus mean predictor\n'
    '(positive gap = temporal rescue exceeds Jensen penalty)',
    fontsize=11
)

# Add text box explaining non-monotone behaviour
axes2[0].text(
    0.98, 0.35,
    'Non-monotone:\n'
    '① below $r_p - r_0$: static-like\n'
    '② above $r_p - r_0$: rescue grows\n'
    '③ large $A_{half}$: off-phase → $r \\approx 0$\n'
    '   extinction dominates → λ falls',
    transform=axes2[0].transAxes, fontsize=8, ha='right', va='bottom',
    bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85)
)

plt.tight_layout()
plt.savefig('fig2_nonmonotone_rescue.pdf', bbox_inches='tight')
plt.show()
print()


# ===========================================================================
#  FIG 3 — Optimal amplitude A_half*(r_0) and persistence gain
#  For each r_0 < r_p, find A_half that maximises λ_product,
#  and compare with static λ (showing the rescue gain).
# ===========================================================================

print("Fig 3: optimal amplitude scan …")
r0_scan   = np.linspace(0.3, r_p - 0.05, 20)
Ah_dense  = np.linspace(0.0, r_p + 1.8, 50)

lp_opt    = np.zeros(len(r0_scan))   # max λ_product over A_half
Ah_opt    = np.zeros(len(r0_scan))   # argmax
lp_stat   = np.zeros(len(r0_scan))   # static λ (A_half = 0)

for j, r_0 in enumerate(r0_scan):
    col = []
    for A_h in Ah_dense:
        scen  = make_centered_scenario(r_0, A_h, w, n, L, e_0, z, beta, SEED)
        supra = build_supra(scen, n, c, T)
        col.append(lambda_product(supra))
    col = np.array(col)
    lp_opt[j]  = col.max()
    Ah_opt[j]  = Ah_dense[col.argmax()]
    lp_stat[j] = col[0]   # A_half = 0
    print(f"  r_0 = {r_0:.2f},  λ_stat = {lp_stat[j]:.3f},  λ_opt = {lp_opt[j]:.3f},  A* = {Ah_opt[j]:.2f}")

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4.5))
fig3.suptitle(
    r'Optimal Amplitude and Rescue Gain vs Mean Radius $r_0$  [$w = 1/T$]',
    fontsize=13
)

# Panel A: λ_static vs λ_optimal
axes3[0].plot(r0_scan, lp_stat, '-o', color='crimson',   ms=4, lw=2, label=r'$\lambda_{static}$ ($A_{half}=0$)')
axes3[0].plot(r0_scan, lp_opt,  '-o', color='seagreen',  ms=4, lw=2, label=r'$\lambda_{product}$ at optimal $A_{half}^*$')
axes3[0].fill_between(r0_scan, lp_stat, lp_opt, alpha=0.15, color='seagreen', label='Rescue gain')
axes3[0].axhline(1, color='k', ls=':', lw=1.2)
axes3[0].axvline(r_c, color='royalblue', ls='--', lw=1.5, label=f'$r_c$={r_c:.2f}')
axes3[0].axvline(r_p, color='crimson',   ls='--', lw=1.5, label=f'$r_p$={r_p:.2f}')
axes3[0].set_xlabel('Mean radius $r_0$', fontsize=12)
axes3[0].set_ylabel(r'Spectral radius $\lambda$', fontsize=11)
axes3[0].set_title(r'Static vs optimal temporal $\lambda$', fontsize=11)
axes3[0].legend(fontsize=8); axes3[0].grid(True, alpha=0.3)

# Panel B: optimal amplitude A_half*(r_0)
axes3[1].plot(r0_scan, Ah_opt, '-o', color='darkorange', ms=4, lw=2)
axes3[1].plot(r0_scan, r_p - r0_scan, '--', color='gray', lw=1.5,
              label=r'$r_p - r_0$ (rescue onset)')
axes3[1].plot(r0_scan, np.maximum(r_c - r0_scan, 0), ':', color='royalblue', lw=1.5,
              label=r'$r_c - r_0$ (percolation onset)')
axes3[1].axvline(r_c, color='royalblue', ls='--', lw=1.5)
axes3[1].axvline(r_p, color='crimson',   ls='--', lw=1.5)
axes3[1].set_xlabel('Mean radius $r_0$', fontsize=12)
axes3[1].set_ylabel(r'Optimal amplitude $A_{half}^*$', fontsize=11)
axes3[1].set_title(r'Optimal $A_{half}^*(r_0)$', fontsize=11)
axes3[1].legend(fontsize=8); axes3[1].grid(True, alpha=0.3)

# Panel C: rescue gain = λ_opt − λ_stat
axes3[2].plot(r0_scan, lp_opt - lp_stat, '-o', color='purple', ms=4, lw=2)
axes3[2].axhline(0, color='k', ls=':', lw=1.2)
axes3[2].axvline(r_c, color='royalblue', ls='--', lw=1.5, label=f'$r_c$={r_c:.2f}')
axes3[2].axvline(r_p, color='crimson',   ls='--', lw=1.5, label=f'$r_p$={r_p:.2f}')
axes3[2].set_xlabel('Mean radius $r_0$', fontsize=12)
axes3[2].set_ylabel(r'Rescue gain $\lambda_{opt} - \lambda_{static}$', fontsize=11)
axes3[2].set_title('Rescue gain vs $r_0$\n(peaks where static is most sub-critical)', fontsize=11)
axes3[2].legend(fontsize=8); axes3[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig3_optimal_amplitude.pdf', bbox_inches='tight')
plt.show()

print("\nAll figures saved as PDF.")
