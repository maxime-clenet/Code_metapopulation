"""
Test_RGN_analysis.py

Persistence of a metapopulation in a dynamically changing landscape —
oscillating Random Geometric Network (RGN).

Central question: what controls whether the time-averaged ('mean-matrix')
landscape is a reliable predictor of persistence, and when does the temporal
structure of connectivity matter?

Key concepts
------------
• λ_product  : ρ(∏ M_k)^{1/T}  — temporal metapopulation capacity (true predictor)
• λ_mean     : ρ(M̄)            — naive mean-landscape predictor
• Duty cycle : φ(A) = fraction of time r(t) > r_c  (analytical, depends only on A)
• Regimes    : fast  (max(c,e) >> w),  intermediate,  slow  (max(c,e) << w)

Experiments
-----------
1. λ vs amplitude A      for several w values
2. λ vs frequency w      for several A values
3. Gap vs c and e₀       (impact of ecological processes; gap ~ c²)
4. Phase diagram (A, w)  with duty-cycle isoclines         [main result]
5. Phase diagrams for three demographic regimes (fast / intermediate / slow)
6. Shape of r(t): sinusoidal vs. square-wave at same mean  (autocorrelation effect)
7. Order randomisation: λ_original vs λ_shuffled vs λ_mean (temporal structure)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap


# ===========================================================================
#  1. GEOMETRY
# ===========================================================================

def periodic_distance(locations, square_size):
    """Euclidean distance matrix with periodic (torus) boundary conditions."""
    delta = np.abs(locations[:, None, :] - locations[None, :, :])
    delta = np.minimum(delta, square_size - delta)
    return np.sqrt((delta ** 2).sum(axis=2))


def percolation_radius(n, square_size):
    """r_c: critical connectivity radius for RGN percolation."""
    return np.sqrt(4.512 / (np.pi * n / square_size ** 2))


def percolation_amplitude(n, square_size):
    """A_c = 2·r_c: static amplitude at which mean r equals r_c."""
    return 2.0 * percolation_radius(n, square_size)


def lattice_positions(n, square_size):
    """Regular square lattice (n must be a perfect square)."""
    side = int(np.round(np.sqrt(n)))
    xs = np.linspace(0, square_size, side, endpoint=False)
    coords = np.array([(x, y) for y in xs for x in xs])
    return coords[:n]


# ===========================================================================
#  2. DUTY CYCLE (analytical)
# ===========================================================================

def duty_cycle(A, r_c):
    """
    Fraction of time r(t) = (A/2)(1+sin(2πwt)) exceeds r_c.

    Depends only on A and r_c (independent of frequency w).
    φ(A) = ½ − arcsin(2·r_c/A − 1) / π
    """
    if A <= r_c:
        return 0.0
    h = float(np.clip(2.0 * r_c / A - 1.0, -1.0, 1.0))
    return 0.5 - np.arcsin(h) / np.pi


def amplitude_for_duty_cycle(phi, r_c):
    """
    Inverse: A such that duty_cycle(A, r_c) = φ.
    A = 2·r_c / (1 + cos(π·φ)).
    """
    if phi <= 0:
        return r_c
    if phi >= 1.0:
        return np.inf
    return 2.0 * r_c / (1.0 + np.cos(np.pi * phi))


# ===========================================================================
#  3. SCENARIO FACTORIES
# ===========================================================================

def _base_geometry(n, square_size, e_0, z, beta, seed, lattice=False):
    """Shared geometry setup used by all scenario factories."""
    if lattice:
        patch_locations = lattice_positions(n, square_size)
        np.random.seed(seed)          # kept for reproducibility of other calls
    else:
        np.random.seed(seed)
        patch_locations = np.random.rand(n, 2) * square_size

    A_std     = np.ones(n) / n
    distances = periodic_distance(patch_locations, square_size)
    np.fill_diagonal(distances, np.inf)
    e_vec = e_0 * A_std ** (-z)
    A_j   = A_std[None, :] ** beta
    return distances, e_vec, A_j


def make_sinusoidal_scenario(A_ampl, w_freq, n, square_size=20,
                              e_0=0.001, z=1.0, beta=1.0, seed=40, lattice=False):
    """
    Sinusoidal threshold:  r(k) = (A/2) · (1 + sin(2π·w·k)).
    Returns  scenario(k) → (S, e).
    """
    distances, e_vec, A_j = _base_geometry(n, square_size, e_0, z, beta, seed, lattice)

    def scenario(k):
        r_k = 0.5 * A_ampl * (1.0 + np.sin(2.0 * np.pi * w_freq * k))
        S   = A_j * (distances < r_k).astype(float)
        np.fill_diagonal(S, 0)
        return S.copy(), e_vec.copy()

    return scenario


def make_squarewave_scenario(A_on, phi_on, w_freq, n, square_size=20,
                              e_0=0.001, z=1.0, beta=1.0, seed=40, lattice=False):
    """
    Square-wave (Heaviside) threshold:
      r(k) = A_on   if (w·k mod 1) < phi_on      ['on'  phase]
      r(k) = 0      otherwise                     ['off' phase]

    Mean r̄ = phi_on · A_on.  For a fair comparison with the sinusoidal case
    (same mean r̄ = A_sin/2), set phi_on = 0.5, A_on = A_sin.
    """
    distances, e_vec, A_j = _base_geometry(n, square_size, e_0, z, beta, seed, lattice)

    def scenario(k):
        phase = (w_freq * k) % 1.0
        r_k   = A_on if phase < phi_on else 0.0
        S     = A_j * (distances < r_k).astype(float)
        np.fill_diagonal(S, 0)
        return S.copy(), e_vec.copy()

    return scenario


# ===========================================================================
#  4. SUPRA-ADJACENCY BUILDER
# ===========================================================================

def build_supra(scenario_func, n, c, T):
    """
    Returns array of shape (T-1, n, n):
      M_k = I − diag(e) + c · S_k,   k = 0, …, T-2.
    """
    supra = np.zeros((T - 1, n, n))
    for k in range(T - 1):
        S, e      = scenario_func(k)
        supra[k]  = np.eye(n) - np.diag(e) + c * S
    return supra


# ===========================================================================
#  5. SPECTRAL METRICS
# ===========================================================================

def lambda_product(supra):
    """
    Temporal metapopulation capacity: ρ(∏ M_k)^{1/(T-1)}.
    λ > 1 → persistence;  λ < 1 → extinction.

    Power iteration with renormalisation avoids overflow/underflow for large T
    or extreme parameter values (e.g. ρ^500 → ∞ without renormalisation).
    Computes  exp((1/T) Σ_k log‖M_k v_k‖) = ρ(∏ M_k)^{1/T}.
    """
    n      = supra.shape[1]
    T      = supra.shape[0]
    v      = np.ones(n) / np.sqrt(n)
    log_s  = 0.0
    for M in supra:
        v    = M @ v
        nrm  = np.linalg.norm(v)
        if nrm < 1e-300:
            return 0.0
        log_s += np.log(nrm)
        v    /= nrm
    return float(np.exp(log_s / T))


def lambda_mean(supra):
    """
    Mean-matrix predictor: ρ(M̄).
    Ignores temporal ordering; equivalent to replacing the landscape by its
    time-average M̄ = (1/T) Σ M_k = I − diag(e) + c · S̄.
    """
    return float(np.max(np.abs(np.linalg.eigvals(np.mean(supra, axis=0)))))


def lambda_product_shuffled(supra, n_shuffles=20, seed=0):
    """
    Average λ_product over random permutations of the matrix sequence.

    Interpretation:
      • λ_shuffled ≈ λ_product  → temporal order irrelevant (commuting regime)
      • λ_shuffled ≠ λ_product  → sequence ordering/autocorrelation matters
      • λ_shuffled → λ_mean     in the large-T i.i.d. limit

    Decomposition:
      λ_original − λ_shuffled  = effect of temporal autocorrelation
      λ_shuffled  − λ_mean     = effect of matrix distribution (non-commutativity)
    """
    rng  = np.random.default_rng(seed)
    idx  = np.arange(len(supra))
    vals = []
    for _ in range(n_shuffles):
        rng.shuffle(idx)
        vals.append(lambda_product(supra[idx]))
    return float(np.mean(vals)), float(np.std(vals))


def avg_commutator_norm(supra, subsample=50):
    """
    Non-commutativity measure: mean Frobenius norm of [M_k, M_{k+1}].
    Note: [M_k, M_{k+1}] = c² [S_k, S_{k+1}]  → scales as c².
    A subsample of consecutive pairs is used for efficiency.
    """
    T    = len(supra)
    step = max(1, T // subsample)
    norms = [
        np.linalg.norm(supra[k] @ supra[k + 1] - supra[k + 1] @ supra[k], 'fro')
        for k in range(0, T - 1, step)
    ]
    return float(np.mean(norms))


# ===========================================================================
#  6. PLOTTING HELPERS
# ===========================================================================

def add_duty_cycle_lines(ax, A_range, r_c,
                          phi_levels=(0.1, 0.25, 0.5, 0.75),
                          axis='x', color='white', lw=1.2, alpha=0.85, fontsize=7):
    """
    Overlay vertical (axis='x') or horizontal (axis='y') lines marking
    the amplitudes A corresponding to given duty-cycle values φ.
    Since φ depends only on A (not w), these are straight lines in (A, w) space.
    """
    for phi in phi_levels:
        A_phi = amplitude_for_duty_cycle(phi, r_c)
        if A_range[0] < A_phi < A_range[1]:
            if axis == 'x':
                ax.axvline(A_phi, color=color, ls=':', lw=lw, alpha=alpha)
                ax.text(A_phi, ax.get_ylim()[1] * 0.97,
                        f'φ={phi:.2f}', color=color, fontsize=fontsize,
                        ha='center', va='top')
            else:
                ax.axhline(A_phi, color=color, ls=':', lw=lw, alpha=alpha)
                ax.text(ax.get_xlim()[1] * 0.98, A_phi,
                        f'φ={phi:.2f}', color=color, fontsize=fontsize,
                        ha='right', va='bottom')


def phase_panel(ax, A_grid, wT_grid, values, title, cbar_label,
                vmin=0.5, vmax=1.5, cmap='RdYlGn', r_c=None, A_c=None):
    """Reusable phase-diagram panel with persistence contour and A_c line."""
    im = ax.pcolormesh(A_grid, wT_grid, values,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
    ax.contour(A_grid, wT_grid, values, levels=[1.0],
               colors='k', linewidths=2)
    if A_c is not None:
        ax.axvline(A_c, color='royalblue', ls='--', lw=2,
                   label=f'$A_c$ = {A_c:.2f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Amplitude $A$', fontsize=11)
    ax.set_ylabel('Oscillation cycles $(w \\cdot T)$', fontsize=10)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label=cbar_label)
    return im


# ===========================================================================
#  7. SHARED PARAMETERS
# ===========================================================================

n      = 100
L      = 20
e_0    = 0.001
z      = 1.0
beta   = 1.0
T      = 500
c_base = 1.0

r_c = percolation_radius(n, L)
A_c = 2.0 * r_c
print(f"Percolation radius r_c = {r_c:.3f},  amplitude A_c = {A_c:.3f}\n")

COLORS   = ['steelblue', 'darkorange', 'seagreen', 'crimson']
W_LABELS = [
    'w = 0  (static)',
    'w = 1/T  (1 cycle)',
    'w = 3/T  (3 cycles)',
    'w = 10/T (10 cycles)',
]


# ===========================================================================
#  EXP 1 — λ vs Amplitude A  (several w)
# ===========================================================================

print("Exp 1: amplitude scan …")
A_values = np.linspace(1.0, 12.0, 25)
w_list   = [0.0, 1 / T, 3 / T, 10 / T]

results_A = {}
for w, label, color in zip(w_list, W_LABELS, COLORS):
    lp_arr, lm_arr, cn_arr = [], [], []
    for A in A_values:
        scen  = make_sinusoidal_scenario(A, w, n, L, e_0, z, beta)
        supra = build_supra(scen, n, c_base, T)
        lp_arr.append(lambda_product(supra))
        lm_arr.append(lambda_mean(supra))
        cn_arr.append(avg_commutator_norm(supra))
    results_A[w] = (np.array(lp_arr), np.array(lm_arr), np.array(cn_arr))
    print(f"  {label} done")

fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle('Effect of Connectivity Amplitude $A$ on Persistence Predictors', fontsize=13)

for w, label, color in zip(w_list, W_LABELS, COLORS):
    lp, lm, cn = results_A[w]
    axes[0].plot(A_values, lp, '-',  color=color, lw=2,   label=f'λ_prod  [{label}]')
    axes[0].plot(A_values, lm, '--', color=color, lw=1.5, alpha=0.7)
    axes[1].plot(A_values, lp - lm, '-o', color=color, ms=3, lw=1.5, label=label)
    axes[2].plot(A_values, cn, '-',  color=color, lw=1.5, label=label)

axes[0].plot([], [], '--', color='grey', lw=1.5, alpha=0.7, label='λ_mean (dashed)')
for ax in axes:
    ax.axvline(A_c, color='gray', ls='--', lw=1.5, label=f'$A_c$ = {A_c:.2f}')
    ax.set_xlabel('Amplitude $A$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.5)

axes[0].axhline(1, color='k', ls=':', lw=1)
axes[1].axhline(0, color='k', ls=':', lw=1)
axes[0].set_ylabel('Spectral radius', fontsize=11)
axes[1].set_ylabel('$λ_{product} − λ_{mean}$', fontsize=11)
axes[2].set_ylabel(r'Avg $\|[M_k, M_{k+1}]\|_F$', fontsize=11)
axes[0].set_title('$λ_{product}$ (—) vs $λ_{mean}$ (– –)', fontsize=11)
axes[1].set_title('Gap: temporal product minus mean predictor', fontsize=11)
axes[2].set_title('Non-commutativity of transition matrices', fontsize=11)

plt.tight_layout()
plt.savefig('fig1_amplitude_effect.pdf', bbox_inches='tight')
plt.show()
print()


# ===========================================================================
#  EXP 2 — λ vs Frequency w  (several A relative to A_c)
# ===========================================================================

print("Exp 2: frequency scan …")
w_values = np.array([k / T for k in range(16)])
A_list   = [A_c * f for f in [0.5, 1.0, 1.5, 2.0]]
A_labels = [f'$A$ = {a:.2f}  ({f}×$A_c$)' for a, f in zip(A_list, [0.5, 1.0, 1.5, 2.0])]

results_w = {}
for A, label in zip(A_list, A_labels):
    lp_arr, lm_arr = [], []
    for w in w_values:
        scen  = make_sinusoidal_scenario(A, w, n, L, e_0, z, beta)
        supra = build_supra(scen, n, c_base, T)
        lp_arr.append(lambda_product(supra))
        lm_arr.append(lambda_mean(supra))
    results_w[A] = (np.array(lp_arr), np.array(lm_arr))
    print(f"  A = {A:.2f} done")

fig2, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Effect of Oscillation Frequency on Persistence Predictors', fontsize=13)

for A, label, color in zip(A_list, A_labels, COLORS):
    lp, lm = results_w[A]
    ax0.plot(w_values * T, lp, '-o',  color=color, ms=4, lw=1.5, label=label)
    ax0.plot(w_values * T, lm, '--^', color=color, ms=4, lw=1.2, alpha=0.7)
    ax1.plot(w_values * T, lp - lm, '-o', color=color, ms=4, lw=1.5, label=label)

ax0.axhline(1, color='k', ls=':', lw=1)
ax1.axhline(0, color='k', ls=':', lw=1)
ax0.plot([], [], '-o',  color='grey', ms=4, label='$λ_{product}$')
ax0.plot([], [], '--^', color='grey', ms=4, alpha=0.7, label='$λ_{mean}$')

for ax in (ax0, ax1):
    ax.set_xlabel('Oscillation cycles  $(w \\cdot T)$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

ax0.set_ylabel('Spectral radius', fontsize=11)
ax0.set_title('$λ_{product}$ (—·) vs $λ_{mean}$ (– –△)', fontsize=11)
ax1.set_ylabel('$λ_{product} − λ_{mean}$', fontsize=11)
ax1.set_title('Gap: Product minus Mean predictor', fontsize=11)

plt.tight_layout()
plt.savefig('fig2_frequency_effect.pdf', bbox_inches='tight')
plt.show()
print()


# ===========================================================================
#  EXP 3 — Gap vs c and e₀  (ecological processes)
# ===========================================================================

print(f"Exp 3: ecological processes  (A = {A_c:.2f} = A_c,  w = 1/T) …")
A_fixed = A_c
w_fixed = 1.0 / T

c_values  = np.linspace(0.1, 4.0, 25)
# Biological constraint: e_i = e_0 · n^z < 1  →  e_0 < 0.01 for n=100, z=1.
# Upper bound 0.008 → e_i_max = 0.8.
e0_values = np.linspace(1e-4, 0.008, 25)

lp_c, lm_c, cn_c = [], [], []
for c in c_values:
    scen  = make_sinusoidal_scenario(A_fixed, w_fixed, n, L, e_0, z, beta)
    supra = build_supra(scen, n, c, T)
    lp_c.append(lambda_product(supra))
    lm_c.append(lambda_mean(supra))
    cn_c.append(avg_commutator_norm(supra))
lp_c, lm_c, cn_c = map(np.array, [lp_c, lm_c, cn_c])
print("  c scan done")

lp_e, lm_e = [], []
for e in e0_values:
    scen  = make_sinusoidal_scenario(A_fixed, w_fixed, n, L, e, z, beta)
    supra = build_supra(scen, n, c_base, T)
    lp_e.append(lambda_product(supra))
    lm_e.append(lambda_mean(supra))
lp_e, lm_e = map(np.array, [lp_e, lm_e])
print("  e₀ scan done")

fig3, axes = plt.subplots(2, 3, figsize=(17, 10))
fig3.suptitle(
    f'Impact of Colonisation $c$ and Extinction $e_0$'
    f'  ($A = A_c = {A_c:.2f}$,  $w = 1/T$)',
    fontsize=13
)

for row, (xvals, lp, lm, xlabel, clr) in enumerate([
    (c_values,  lp_c, lm_c, 'Colonisation rate $c$', 'seagreen'),
    (e0_values, lp_e, lm_e, 'Extinction rate $e_0$',  'crimson'),
]):
    axes[row, 0].plot(xvals, lp, '-o',  color='steelblue',  ms=4, label='$λ_{product}$')
    axes[row, 0].plot(xvals, lm, '--s', color='darkorange', ms=4, label='$λ_{mean}$')
    axes[row, 0].axhline(1, color='k', ls=':', lw=1)
    axes[row, 0].set_xlabel(xlabel, fontsize=12)
    axes[row, 0].set_ylabel('Spectral radius', fontsize=11)
    axes[row, 0].set_title(f'λ vs {xlabel}', fontsize=11)
    axes[row, 0].legend(); axes[row, 0].grid(True, alpha=0.3)

    axes[row, 1].plot(xvals, lp - lm, '-o', color=clr, ms=4)
    axes[row, 1].axhline(0, color='k', ls=':', lw=1)
    axes[row, 1].set_xlabel(xlabel, fontsize=12)
    axes[row, 1].set_ylabel('$λ_{product} − λ_{mean}$', fontsize=11)
    axes[row, 1].set_title(f'Gap vs {xlabel}', fontsize=11)
    axes[row, 1].grid(True, alpha=0.3)

axes[0, 1].annotate(
    r'$[M_k, M_{k+1}] = c^2 [S_k, S_{k+1}]$' '\n' r'$\Rightarrow$ gap $\sim c^2$',
    xy=(0.04, 0.82), xycoords='axes fraction', fontsize=10,
    bbox=dict(boxstyle='round', fc='wheat', alpha=0.8)
)

axes[0, 2].scatter(cn_c, lp_c - lm_c, c=c_values, cmap='plasma', s=40, zorder=3)
sm = plt.cm.ScalarMappable(cmap='plasma',
                           norm=plt.Normalize(c_values.min(), c_values.max()))
plt.colorbar(sm, ax=axes[0, 2], label='$c$')
axes[0, 2].axhline(0, color='k', ls=':', lw=1)
axes[0, 2].set_xlabel(r'Non-commutativity $\|[M_k, M_{k+1}]\|_F$', fontsize=10)
axes[0, 2].set_ylabel('Gap  $λ_{product} − λ_{mean}$', fontsize=11)
axes[0, 2].set_title('Gap vs Non-commutativity ($c$ scan)', fontsize=11)
axes[0, 2].grid(True, alpha=0.3)

axes[1, 2].scatter(e0_values, lp_e - lm_e, c=e0_values, cmap='coolwarm', s=40, zorder=3)
sm2 = plt.cm.ScalarMappable(cmap='coolwarm',
                             norm=plt.Normalize(e0_values.min(), e0_values.max()))
plt.colorbar(sm2, ax=axes[1, 2], label='$e_0$')
axes[1, 2].axhline(0, color='k', ls=':', lw=1)
axes[1, 2].set_xlabel('Extinction rate $e_0$', fontsize=12)
axes[1, 2].set_ylabel('Gap  $λ_{product} − λ_{mean}$', fontsize=11)
axes[1, 2].set_title('Gap vs Extinction rate', fontsize=11)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig3_process_effects.pdf', bbox_inches='tight')
plt.show()
print()


# ===========================================================================
#  EXP 4 — Phase diagram (A, w·T) with duty-cycle isoclines  [main result]
#  n=50, T=200 for computational tractability.
#
#  Agreement map:
#    0 = both predict extinction
#    1 = λ_mean > 1 but λ_product < 1  (mean over-predicts → false safety)
#    2 = λ_product > 1 but λ_mean < 1  (mean under-predicts → false alarm)
#    3 = both predict persistence
# ===========================================================================

print("Exp 4: phase diagram with duty-cycle isoclines …")
n_ph   = 50
T_ph   = 200
L_ph   = 20
r_c_ph = percolation_radius(n_ph, L_ph)
A_c_ph = 2.0 * r_c_ph

A_grid_ph = np.linspace(1.0, 12.0, 20)
w_grid_ph = np.array([k / T_ph for k in range(16)])
wT_ph     = w_grid_ph * T_ph

lp_mat = np.zeros((len(w_grid_ph), len(A_grid_ph)))
lm_mat = np.zeros_like(lp_mat)

for i, w_val in enumerate(w_grid_ph):
    for j, A_val in enumerate(A_grid_ph):
        scen          = make_sinusoidal_scenario(A_val, w_val, n_ph, L_ph, e_0, z, beta, seed=40)
        supra         = build_supra(scen, n_ph, c_base, T_ph)
        lp_mat[i, j]  = lambda_product(supra)
        lm_mat[i, j]  = lambda_mean(supra)
    print(f"  w row {i+1}/{len(w_grid_ph)} done")

agreement = (lp_mat > 1).astype(int) + 2 * (lm_mat > 1).astype(int)

# Duty-cycle values for isocline overlay
phi_iso   = [0.1, 0.25, 0.5, 0.75]
A_iso     = [amplitude_for_duty_cycle(p, r_c_ph) for p in phi_iso]

cmap_agree = ListedColormap(['#d73027', '#fc8d59', '#91cf60', '#1a9641'])

fig4, axes = plt.subplots(1, 3, figsize=(17, 5))
fig4.suptitle(
    f'Phase Diagram: Persistence in $(A,\\ w·T)$ Space  '
    f'[$n$={n_ph}, $T$={T_ph}, $c$={c_base}, $e_0$={e_0}]',
    fontsize=12
)

kw = dict(shading='nearest')

phase_panel(axes[0], A_grid_ph, wT_ph, lp_mat, r'$\lambda_{product}$  (true predictor)',
            r'$\lambda_{product}$', r_c=r_c_ph, A_c=A_c_ph)
phase_panel(axes[1], A_grid_ph, wT_ph, lm_mat, r'$\lambda_{mean}$  (naive predictor)',
            r'$\lambda_{mean}$', r_c=r_c_ph, A_c=A_c_ph)

im2 = axes[2].pcolormesh(A_grid_ph, wT_ph, agreement,
                          cmap=cmap_agree, vmin=-0.5, vmax=3.5, **kw)
axes[2].contour(A_grid_ph, wT_ph, lp_mat, levels=[1.0], colors='k', linewidths=2, linestyles='-')
axes[2].contour(A_grid_ph, wT_ph, lm_mat, levels=[1.0], colors='k', linewidths=1.2, linestyles='--')
axes[2].axvline(A_c_ph, color='royalblue', ls='--', lw=2, label=f'$A_c$={A_c_ph:.2f}')
axes[2].legend(fontsize=9)
cbar2 = plt.colorbar(im2, ax=axes[2], ticks=[0, 1, 2, 3])
cbar2.ax.set_yticklabels(['Both extinct', 'Mean over-predicts\n(false safety)',
                           'Mean under-predicts\n(false alarm)', 'Both persist'])
axes[2].set_title('Prediction agreement  (— $λ_{prod}$=1,  – – $λ_{mean}$=1)', fontsize=10)

# Overlay duty-cycle isoclines on all panels
for ax in axes:
    for phi, A_phi in zip(phi_iso, A_iso):
        if A_grid_ph[0] < A_phi < A_grid_ph[-1]:
            ax.axvline(A_phi, color='white', ls=':', lw=1.2, alpha=0.85)
            ax.text(A_phi, wT_ph[-1] * 0.95, f'φ={phi}',
                    color='white', fontsize=6.5, ha='center', va='top')
    ax.set_xlabel('Amplitude $A$', fontsize=11)
    ax.set_ylabel('Oscillation cycles $(w·T)$', fontsize=10)

plt.tight_layout()
plt.savefig('fig4_phase_diagram.pdf', bbox_inches='tight')
plt.show()
print()


# ===========================================================================
#  EXP 5 — Three demographic regimes: fast / intermediate / slow
#  Regime defined by  max(c, e_i) / w  where  e_i = e_0 · n.
#  Fast : max(c,e_i) >> w,   Slow : max(c,e_i) << w.
#  Three panels show (A, w·T) phase diagram for λ_product, each at different (c, e_0).
# ===========================================================================

print("Exp 5: three regime phase diagrams …")

REGIMES = {
    'Fast\n$\\max(c,e_i) \\gg w$':  dict(c=2.0,  e_0=0.002),   # e_i = 0.2, max=2.0
    'Intermediate\n$\\max(c,e_i) \\sim w$': dict(c=0.2,  e_0=0.0002),  # e_i = 0.02, max=0.2
    'Slow\n$\\max(c,e_i) \\ll w$':  dict(c=0.02, e_0=2e-5),    # e_i = 0.002, max=0.02
}
# For w_max = 15/200 = 0.075:
#   Fast: max/w_max = 2/0.075 ≈ 27  (fast ✓)
#   Intermediate: 0.2/0.075 ≈ 2.7  (borderline)
#   Slow: 0.02/0.075 ≈ 0.27 (slow ✓)

fig5, axes = plt.subplots(1, 3, figsize=(17, 5))
fig5.suptitle(
    'Phase Diagram for Three Demographic Regimes\n'
    r'(coloured by $\lambda_{product}$, black contour = persistence threshold)',
    fontsize=12
)

for ax, (regime_label, params) in zip(axes, REGIMES.items()):
    lp_reg = np.zeros((len(w_grid_ph), len(A_grid_ph)))
    c_r    = params['c']
    e0_r   = params['e_0']
    ei_r   = e0_r * n_ph        # effective extinction rate per patch
    for i, w_val in enumerate(w_grid_ph):
        for j, A_val in enumerate(A_grid_ph):
            scen         = make_sinusoidal_scenario(A_val, w_val, n_ph, L_ph, e0_r, z, beta, seed=40)
            supra        = build_supra(scen, n_ph, c_r, T_ph)
            lp_reg[i, j] = lambda_product(supra)

    phase_panel(ax, A_grid_ph, wT_ph, lp_reg,
                f'{regime_label}\n$c$={c_r}, $e_i$={ei_r:.3f}',
                r'$\lambda_{product}$', r_c=r_c_ph, A_c=A_c_ph)

    # Duty-cycle isoclines
    for phi, A_phi in zip(phi_iso, A_iso):
        if A_grid_ph[0] < A_phi < A_grid_ph[-1]:
            ax.axvline(A_phi, color='white', ls=':', lw=1.2, alpha=0.85)
            ax.text(A_phi, wT_ph[-1] * 0.95, f'φ={phi}',
                    color='white', fontsize=6.5, ha='center', va='top')

    print(f"  {regime_label.split(chr(10))[0]} done")

plt.tight_layout()
plt.savefig('fig5_regimes.pdf', bbox_inches='tight')
plt.show()
print()


# ===========================================================================
#  EXP 6 — Shape of r(t): sinusoidal vs. square wave
#  Same mean r̄ = A/2, same amplitude range [0, A], different time distribution.
#
#  Key question: does persistence depend only on the mean (→ same λ),
#  or on the shape / temporal autocorrelation structure (→ different λ)?
#
#  Duty cycle comparison (both have same mean r̄ = A/2):
#    Square wave:  φ_sw = 0.5  whenever A > r_c  (abrupt switch, symmetric)
#    Sinusoidal:   φ_sin(A) = ½ − arcsin(2r_c/A − 1)/π
#    Difference:   Δφ = φ_sw − φ_sin > 0  for A < 2r_c  (square wave spends more
#                  time above r_c),  Δφ < 0  for A > 2r_c.
# ===========================================================================

print("Exp 6: sinusoidal vs. square-wave shape comparison …")
A_shape  = np.linspace(1.0, 12.0, 20)
w_shape_list   = [1 / T, 3 / T, 10 / T]
w_shape_labels = ['$w = 1/T$', '$w = 3/T$', '$w = 10/T$']

results_shape = {}
for w, label in zip(w_shape_list, w_shape_labels):
    lp_sin, lp_sq = [], []
    dc_sin, dc_sq = [], []
    for A in A_shape:
        # Sinusoidal
        scen_sin  = make_sinusoidal_scenario(A, w, n, L, e_0, z, beta)
        supra_sin = build_supra(scen_sin, n, c_base, T)
        lp_sin.append(lambda_product(supra_sin))

        # Square wave: same mean r̄ = A/2 → phi_on=0.5, A_on=A
        scen_sq  = make_squarewave_scenario(A, 0.5, w, n, L, e_0, z, beta)
        supra_sq = build_supra(scen_sq, n, c_base, T)
        lp_sq.append(lambda_product(supra_sq))

        # Analytical duty cycles (both have mean r̄ = A/2)
        dc_sin.append(duty_cycle(A, r_c))
        dc_sq.append(0.5 if A > r_c else 0.0)

    results_shape[w] = {
        'lp_sin': np.array(lp_sin), 'lp_sq': np.array(lp_sq),
        'dc_sin': np.array(dc_sin), 'dc_sq': np.array(dc_sq),
    }
    print(f"  {label} done")

fig6, axes = plt.subplots(1, 3, figsize=(17, 5))
fig6.suptitle(
    r'Shape of $r(t)$: Sinusoidal vs. Square Wave  (same mean $\bar{r} = A/2$)',
    fontsize=13
)

for ax, w, label, color in zip(axes, w_shape_list, w_shape_labels, COLORS[:3]):
    res = results_shape[w]
    ax.plot(A_shape, res['lp_sin'], '-o', color=color, ms=4, lw=2,   label='Sinusoidal')
    ax.plot(A_shape, res['lp_sq'],  '--s', color=color, ms=4, lw=1.5, alpha=0.8, label='Square wave')
    ax.axhline(1, color='k', ls=':', lw=1)
    ax.axvline(A_c, color='gray', ls='--', lw=1.5, label=f'$A_c$={A_c:.2f}')

    # Shade area between duty-cycle curves to highlight Δφ
    ax2 = ax.twinx()
    ax2.fill_between(A_shape, res['dc_sq'], res['dc_sin'],
                     where=(res['dc_sq'] >= res['dc_sin']),
                     color='blue', alpha=0.12, label='Δφ > 0 (sq > sin)')
    ax2.fill_between(A_shape, res['dc_sq'], res['dc_sin'],
                     where=(res['dc_sq'] < res['dc_sin']),
                     color='red', alpha=0.12, label='Δφ < 0 (sin > sq)')
    ax2.plot(A_shape, res['dc_sin'], ':', color='navy',  lw=1.2, alpha=0.6, label='φ sinusoidal')
    ax2.plot(A_shape, res['dc_sq'],  ':',  color='teal',  lw=1.2, alpha=0.6, label='φ square')
    ax2.set_ylabel('Duty cycle φ', fontsize=9, color='navy')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor='navy', labelsize=8)
    ax2.legend(fontsize=7, loc='lower right')

    ax.set_xlabel('Amplitude $A$', fontsize=12)
    ax.set_ylabel('$λ_{product}$', fontsize=11)
    ax.set_title(f'{label}', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig6_shape_comparison.pdf', bbox_inches='tight')
plt.show()
print()


# ===========================================================================
#  EXP 7 — Order randomisation: λ_original vs λ_shuffled vs λ_mean
#  Decomposes the temporal-structure effect:
#    λ_original − λ_shuffled  = contribution of temporal autocorrelation
#    λ_shuffled  − λ_mean     = contribution of matrix distribution (non-commutativity)
#    λ_original  − λ_mean     = total gap (previous experiments)
# ===========================================================================

print("Exp 7: order randomisation …")
A_rand   = np.linspace(1.0, 12.0, 20)
w_rand_list   = [1 / T, 3 / T, 10 / T]
w_rand_labels = ['$w = 1/T$', '$w = 3/T$', '$w = 10/T$']

results_rand = {}
for w, label in zip(w_rand_list, w_rand_labels):
    lp_orig, lm_arr, lp_shuf, lp_shuf_std = [], [], [], []
    for A in A_rand:
        scen  = make_sinusoidal_scenario(A, w, n, L, e_0, z, beta)
        supra = build_supra(scen, n, c_base, T)
        lp_orig.append(lambda_product(supra))
        lm_arr.append(lambda_mean(supra))
        mu, sigma = lambda_product_shuffled(supra, n_shuffles=15, seed=42)
        lp_shuf.append(mu)
        lp_shuf_std.append(sigma)
    results_rand[w] = {
        'lp':       np.array(lp_orig),
        'lm':       np.array(lm_arr),
        'lp_shuf':  np.array(lp_shuf),
        'lp_std':   np.array(lp_shuf_std),
    }
    print(f"  {label} done")

fig7, axes = plt.subplots(2, 3, figsize=(17, 10))
fig7.suptitle(
    'Order Randomisation: Decomposing Temporal Structure vs. Matrix Distribution',
    fontsize=13
)

for col, (w, label, color) in enumerate(zip(w_rand_list, w_rand_labels, COLORS[:3])):
    res = results_rand[w]
    lp, lm, ls, ss = res['lp'], res['lm'], res['lp_shuf'], res['lp_std']

    # Top row: λ values
    axes[0, col].plot(A_rand, lp, '-o',  color=color, ms=4, lw=2,   label='$λ_{product}$ (original order)')
    axes[0, col].plot(A_rand, ls, '-^',  color=color, ms=4, lw=1.5, alpha=0.6, linestyle='-.', label='$λ_{shuffled}$ (random order)')
    axes[0, col].plot(A_rand, lm, '--s', color='gray', ms=4, lw=1.5, alpha=0.7, label='$λ_{mean}$ (average landscape)')
    axes[0, col].fill_between(A_rand, ls - ss, ls + ss, color=color, alpha=0.12)
    axes[0, col].axhline(1, color='k', ls=':', lw=1)
    axes[0, col].axvline(A_c, color='gray', ls='--', lw=1.5)
    axes[0, col].set_xlabel('Amplitude $A$', fontsize=11)
    axes[0, col].set_ylabel('Spectral radius', fontsize=11)
    axes[0, col].set_title(f'{label}', fontsize=11)
    axes[0, col].legend(fontsize=7.5)
    axes[0, col].grid(True, alpha=0.3)

    # Bottom row: decomposed gaps
    gap_total  = lp - lm         # total gap
    gap_order  = lp - ls         # autocorrelation contribution
    gap_dist   = ls - lm         # distribution contribution

    axes[1, col].plot(A_rand, gap_total, '-o',  color=color, ms=3, lw=2,   label='Total gap  $(λ_{prod} - λ_{mean})$')
    axes[1, col].plot(A_rand, gap_order, '-^',  color='darkorange', ms=3, lw=1.5, label='Autocorrelation  $(λ_{prod} - λ_{shuf})$')
    axes[1, col].plot(A_rand, gap_dist,  '--s', color='seagreen',   ms=3, lw=1.5, label='Distribution  $(λ_{shuf} - λ_{mean})$')
    axes[1, col].axhline(0, color='k', ls=':', lw=1)
    axes[1, col].axvline(A_c, color='gray', ls='--', lw=1.5)
    axes[1, col].set_xlabel('Amplitude $A$', fontsize=11)
    axes[1, col].set_ylabel('Gap', fontsize=11)
    axes[1, col].set_title(f'Gap decomposition — {label}', fontsize=11)
    axes[1, col].legend(fontsize=7.5)
    axes[1, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig7_order_randomisation.pdf', bbox_inches='tight')
plt.show()

print("\nAll experiments complete.  Figures saved as PDF.")
