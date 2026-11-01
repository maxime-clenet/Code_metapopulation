"""
Oscillating random geometric network on a torus, r(k) = (A/2)(1 + sin 2πwk).

Part 1: λ_product vs c, vs A and vs w.
Part 2: λ_product vs A for a static (w = 0) and a slowly oscillating (w = 1/T) network.

CONFIG selects the landscape: 'L10' (former Test_RGN.py) or 'L20' (former Test_RGN_17_06.py).
"""

import matplotlib.pyplot as plt
import numpy as np

from metapop import (build_supra, lambda_product, make_landscape, percolation_radius,
                     sinusoidal_radius, threshold_scenario)

CONFIGS = {
    'L10': dict(L=10, seed=42, e_0=0.002, A_compare=np.linspace(0.5, 6, 10)),
    'L20': dict(L=20, seed=40, e_0=0.001, A_compare=np.linspace(0.5, 8, 20)),
}
CONFIG = 'L10'
RUN_PART_1 = True

n = 100
z = 1
beta = 1
T = 500
L, seed, e_0, A_compare = (CONFIGS[CONFIG][k] for k in ('L', 'seed', 'e_0', 'A_compare'))

land = make_landscape(n, L, seed)


def lam(A, w, c):
    scenario = threshold_scenario(land, sinusoidal_radius(A, w), e_0, z, beta)
    return lambda_product(build_supra(scenario, c, T))


# === PART 1 === #
if RUN_PART_1:
    colonisation_rates = np.linspace(0.1, 2, 20)
    A_values = np.linspace(0, 0.5, 20)
    w_values = np.arange(11) / T

    eig_c = [lam(4.0, 0.0, c) for c in colonisation_rates]
    eig_A = [lam(A, 1 / T, 1.0) for A in A_values]
    eig_w = [lam(4.0, w, 1.0) for w in w_values]

    fig, axs = plt.subplots(3, 1, figsize=(10, 16))
    for ax, x, y, marker, color, title, xlabel in [
        (axs[0], colonisation_rates, eig_c, 'o', None, "Colonisation Rate (c)", "Colonisation rate c"),
        (axs[1], A_values, eig_A, 's', 'green', "Amplitude (A)", "Amplitude A of threshold oscillation"),
        (axs[2], w_values, eig_w, '^', 'purple', "Frequency (w)", "Frequency w of threshold oscillation"),
    ]:
        ax.plot(x, y, marker=marker, color=color)
        ax.set_title(f"Temporal Metapopulation Capacity vs {title}", fontsize=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\lambda_{\hat{M}}$")
        ax.grid(True)
    plt.tight_layout()
    plt.show()


# === PART 2: w = 0 vs w = 1/T === #
c = 1.0
eig_w0 = [lam(A, 0.0, c) for A in A_compare]
eig_w1 = [lam(A, 1 / T, c) for A in A_compare]
A_threshold = 2 * percolation_radius(n, L)     # mean radius A/2 equals r_c

plt.figure(figsize=(8, 5))
plt.plot(A_compare, eig_w0, marker='o', label='w = 0')
plt.plot(A_compare, eig_w1, marker='s', label='w = 1/T')
plt.axvline(A_threshold, color='red', linestyle='--', label=f'2× percolation radius ≈ {A_threshold:.2f}')
plt.xlabel("Amplitude of seasonal fluctuations A")
plt.ylabel(r"Temporal metapopulation capacity $\lambda_{\hat{M}}$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
