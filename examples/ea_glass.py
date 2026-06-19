"""
Edwards-Anderson (EA) Spin Glass — Temperature Sweep with Overlap Analysis.

This script simulates the EA model (nearest-neighbour Ising spins with quenched
random couplings J_ij ∈ {-J, +J}) in 3D for various lattice lengths.

For each lattice length it:
  1. Sweeps inverse temperature β.
  2. Records magnetization evolution, overlap distribution P(q), and
     the spin-glass susceptibility χ_SG = N · ⟨q²⟩.
  3. Plots four panels per lattice length:
       (a) magnetization traces   (b) ⟨q²⟩ vs T
       (c) χ_SG vs T              (d) P(q) kernel-density estimates

Physical expectations
---------------------
- 3D EA:  T_c ≈ 1.1 J (binary ±J couplings).
          χ_SG should show a clear peak near T_c.
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import cast

from spin_engine.models import EdwardsAndersonSystem
from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.interactions.standard import BinaryRandomInteraction
from spin_engine.dynamics import MetropolisHastings
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.scalars import Energy, Magnetization
from spin_engine.measurements.correlations import OverlapDistribution


def generate_betas(
    num_betas: int = 25,
    critical_beta: float = 0.909,
    min_beta: float = 0.2,
    max_beta: float = 2.5,
) -> list[float]:
    """Generates a list of betas concentrated around the critical temperature."""
    dense_range = 0.15
    num_dense = int(0.6 * num_betas)
    num_sparse = num_betas - num_dense

    betas_dense = np.linspace(
        critical_beta - dense_range,
        critical_beta + dense_range,
        num_dense
    )
    lower_tail = np.linspace(
        min_beta,
        critical_beta - dense_range - 0.05,
        num_sparse // 2
    )
    upper_tail = np.linspace(
        critical_beta + dense_range + 0.05,
        max_beta,
        num_sparse - len(lower_tail)
    )

    betas = np.concatenate([lower_tail, betas_dense, upper_tail])
    betas = np.sort(np.unique(betas))
    return betas.tolist()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_ea_sweep(
    lattice_length: int,
    lattice_dim: int,
    betas: list[float],
    lattice_replicas: int = 64,
    J: float = 1.0,
    sweep_length: int = 10000,
    granularity: int = 100,
    coupling_seed: int = 42,
):
    """Run EA model across *betas* for a single disorder realisation."""
    num_flips = cast(tf.Tensor, tf.constant(1))

    # Build quenched coupling matrix (same for all β)
    nn_mask = PeriodicNearestNeighborInteraction().generate(lattice_dim, lattice_length)
    random_J = BinaryRandomInteraction(J=J, seed=coupling_seed).generate(lattice_dim, lattice_length)
    interaction_matrix = nn_mask * random_J

    N = lattice_length ** lattice_dim
    # Number of Monte Carlo sweeps to perform
    sweeps = 5000 
    sweep_length = sweeps * N
    
    print(f"  EA: L={lattice_length}, D={lattice_dim}, N={N}, "
          f"replicas={lattice_replicas}, sweeps={sweeps} (total steps={sweep_length})")

    results: dict = {}

    # Initialize ONCE outside the loop for Simulated Annealing (hot to cold)
    system = EdwardsAndersonSystem(
        lattice_length=lattice_length,
        lattice_dim=lattice_dim,
        lattice_replicas=lattice_replicas,
        interaction_matrix=interaction_matrix,
        initial_magnetization=0.0, # Hot state (T=infinity)
    )
    sim = MetropolisHastings(system)

    tracker = Tracker(
        measurements=[
            Energy(system),
            Magnetization(system),
            OverlapDistribution(system),
        ],
        granularity=granularity,
    )

    # Ascending betas = decreasing temperature (Annealing)
    for beta in betas:
        print(f"    β = {beta:.4f}")

        sim.sweep(
            tracker=tracker,
            beta=tf.constant(beta, dtype=tf.float32),
            num_disturbances=num_flips,
            sweep_length=sweep_length,
        )

        beta_data: dict = {}
        for name, var in tracker.history.items():
            beta_data[name] = var.numpy()
        results[beta] = beta_data

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _equilibrium_q2(results: dict, betas: list[float]):
    q2_means, q2_stds = [], []
    for beta in betas:
        pq = results[beta]["OverlapDistribution"]
        half = len(pq) // 2
        q_flat = pq[half:]
        q2_per_step = np.mean(q_flat ** 2, axis=1)
        q2_means.append(np.mean(q2_per_step))
        q2_stds.append(np.std(q2_per_step))
    return np.array(q2_means), np.array(q2_stds)


def plot_ea_results(
    all_results: dict[int, dict],
    lengths: list[int],
    betas: list[float],
    N_per_length: dict[int, int],
    J: float = 1.0,
):
    n_lengths = len(lengths)
    fig, axes = plt.subplots(n_lengths, 4, figsize=(22, 5 * n_lengths))
    if n_lengths == 1:
        axes = axes[np.newaxis, :]

    temps = np.array([1.0 / b for b in betas])

    # Known / expected critical temperatures for EA ±J in 3D
    Tc_expected = 1.1

    for row, L in enumerate(lengths):
        results = all_results[L]
        N = N_per_length[L]
        q2_means, q2_stds = _equilibrium_q2(results, betas)

        # (a) Magnetization traces (highest β)
        ax_mag = axes[row, 0]
        last_beta = betas[-1]
        mag = results[last_beta]["Magnetization"]
        steps = np.arange(mag.shape[0]) * 100
        for r in range(min(10, mag.shape[1])):
            ax_mag.plot(steps, mag[:, r], alpha=0.3, linewidth=0.8)
        ax_mag.plot(steps, np.mean(mag, axis=1), "k--", lw=2, label="Mean")
        ax_mag.set_title(f"EA 3D L={L}  Magnetization (β={last_beta})")
        ax_mag.set_xlabel("MC step")
        ax_mag.set_ylabel("m")
        ax_mag.legend(fontsize=8)
        ax_mag.grid(alpha=0.3)

        # (b) ⟨q²⟩ vs T
        ax_q2 = axes[row, 1]
        ax_q2.errorbar(temps, q2_means, yerr=q2_stds,
                       fmt="o-", color="crimson", capsize=4, markersize=5)
        ax_q2.axvline(x=Tc_expected, ls="--", color="grey",
                      label=rf"$T_c \approx {Tc_expected}$")
        ax_q2.set_xlabel(r"$T = 1/\beta$")
        ax_q2.set_ylabel(r"$\langle q^2 \rangle$")
        ax_q2.set_title(f"EA 3D L={L}  Order Parameter")
        if ax_q2.get_legend_handles_labels()[1]:
            ax_q2.legend(fontsize=8)
        ax_q2.grid(alpha=0.3)

        # (c) χ_SG = N ⟨q²⟩
        ax_chi = axes[row, 2]
        chi = N * q2_means
        chi_err = N * q2_stds
        ax_chi.errorbar(temps, chi, yerr=chi_err,
                        fmt="s-", color="teal", capsize=4, markersize=5)
        peak_idx = np.argmax(chi)
        ax_chi.axvline(x=temps[peak_idx], ls=":", color="red",
                       label=rf"peak $T \approx {temps[peak_idx]:.2f}$")
        ax_chi.axvline(x=Tc_expected, ls="--", color="grey",
                       label=rf"$T_c \approx {Tc_expected}$")
        ax_chi.set_xlabel(r"$T$")
        ax_chi.set_ylabel(r"$\chi_{\rm SG} = N\,\langle q^2\rangle$")
        ax_chi.set_title(f"EA 3D L={L}  Susceptibility")
        ax_chi.legend(fontsize=8)
        ax_chi.grid(alpha=0.3)

        # (d) P(q)
        ax_pq = axes[row, 3]
        targets = [0.5, 0.8, 1.0, 1.5, 2.5]
        all_betas = sorted(list(results.keys()))
        selected = []
        for target in targets:
            closest_beta = min(all_betas, key=lambda b: abs(b - target))
            if closest_beta not in selected:
                selected.append(closest_beta)
        palette = sns.color_palette("coolwarm", len(selected))
        for i, beta in enumerate(selected):
            pq = results[beta]["OverlapDistribution"]
            half = len(pq) // 2
            q_vals = pq[half:].flatten()
            sns.kdeplot(q_vals, ax=ax_pq, color=palette[i], fill=True,
                        alpha=0.2, label=f"T={1/beta:.2f}")
        ax_pq.set_xlabel(r"$q$")
        ax_pq.set_ylabel(r"$P(q)$")
        ax_pq.set_title(f"EA 3D L={L}  Overlap Distribution")
        ax_pq.legend(fontsize=8)
        ax_pq.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("examples/images", exist_ok=True)
    path = "examples/images/ea_glass_overlap.png"
    plt.savefig(path, dpi=150)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Concentration of β near expected 3D transition (Tc ≈ 1.1 => beta_c ≈ 0.909)
    betas = generate_betas(num_betas=10, critical_beta=0.909, min_beta=0.2, max_beta=2.5)
    J = 1.0
    D = 3

    lengths = [4, 8, 16]
    N_per_length = {L: L**D for L in lengths}
    all_results: dict[int, dict] = {}

    for L in lengths:
        print(f"\n=== EA  D={D}  L={L}  N={L**D} ===")
        all_results[L] = run_ea_sweep(
            lattice_length=L,
            lattice_dim=D,
            betas=betas,
            lattice_replicas=64,
            J=J,
            sweep_length=8000,
            granularity=100,
        )

    plot_ea_results(all_results, lengths, betas, N_per_length, J=J)


if __name__ == "__main__":
    main()
