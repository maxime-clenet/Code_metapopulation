from pathlib import Path

import matplotlib.pyplot as plt

from .spectral import threshold_crossing

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def savefig(name, fig=None):
    """Save to figures/<name> at the repository root."""
    FIG_DIR.mkdir(exist_ok=True)
    (fig or plt.gcf()).savefig(FIG_DIR / name, dpi=200, bbox_inches='tight')


def plot_persistence_vs_c(ax, c_values, lambdas, occupancy=None, label='', color=None):
    """λ_product and final mean occupancy vs c, with the interpolated λ = 1 threshold."""
    line, = ax.plot(c_values, lambdas, marker='o', color=color,
                    label=f'Temporal metapopulation capacity {label}'.strip())
    if occupancy is not None:
        ax.plot(c_values, occupancy, marker='x', ls='--', color=line.get_color(),
                label=f'Mean final occupancy {label}'.strip())
    c_star = threshold_crossing(c_values, lambdas)
    if c_star is None:
        print(f"No crossing of λ = 1 in the colonisation range {label}")
    else:
        ax.axvline(c_star, color=line.get_color(), ls=':',
                   label=f'λ = 1 at c ≈ {c_star:.3f} {label}'.strip())
    ax.axhline(1, color='grey', ls='--', lw=1)
    ax.set_xlabel('Colonisation rate c', fontsize=14)
    ax.grid(True)
    return c_star


def plot_occupancy(P, title, groups=None):
    """Occupancy trajectories; `groups` = list of (index_array, color, label)."""
    plt.figure(figsize=(10, 6))
    if groups is None:
        plt.plot(P.T, lw=0.8)
    else:
        for idx, color, label in groups:
            lines = plt.plot(P[idx].T, color=color, lw=0.8)
            lines[0].set_label(label)
        plt.legend(loc='upper right', fontsize=15)
    plt.xlabel('Time step', fontsize=15)
    plt.ylabel('Occupancy probability', fontsize=15)
    plt.title(title, fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
