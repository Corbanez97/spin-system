import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt
from typing import cast, List

from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.models import IsingSystem
from spin_engine.dynamics import MetropolisHastings
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.scalars import Energy, Magnetization, MagneticSusceptibility


def generate_betas(num_betas: int = 12, critical_beta: float = 0.44068) -> List[float]:
    """
    Generates a list of betas concentrated around the critical temperature.
    """
    dense_range = 0.1
    num_dense = int(0.6 * num_betas)
    num_sparse = num_betas - num_dense

    # Dense points
    betas_dense = np.linspace(critical_beta - dense_range,
                              critical_beta + dense_range,
                              num_dense)
    lower_tail = np.linspace(0.1, critical_beta -
                             dense_range - 0.05, num_sparse // 2)
    upper_tail = np.linspace(
        critical_beta + dense_range + 0.05, 1.0, num_sparse - len(lower_tail))

    betas = np.concatenate([lower_tail, betas_dense, upper_tail])
    betas = np.sort(np.unique(betas))

    return betas.tolist()


def run_simulation():
    lattice_dim = 2
    lattice_length = 32
    lattice_replicas = 64
    # Number of sites N = L^2
    num_sites = lattice_length ** 2

    interaction_matrix = PeriodicNearestNeighborInteraction().generate(
        lattice_dim, lattice_length)

    granularity = 100
    num_flips = cast(tf.Tensor, tf.constant(1))
    sweep_length = 100000
    equilibration_steps = sweep_length * granularity // 2

    betas = generate_betas(20)
    results = {}
    susceptibility_results = []

    print(f"Starting simulation with {len(betas)} betas: {betas}")

    for beta in betas:
        print(f"Running for beta={beta:.4f}")

        ising_system = IsingSystem(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix,
            initial_magnetization=.5
        )
        simulation = MetropolisHastings(ising_system)

        tracker = Tracker(measurements=[
            Energy(ising_system),
            Magnetization(ising_system)
        ], granularity=granularity)

        simulation.sweep(
            tracker=tracker,
            beta=beta,
            num_disturbances=num_flips,
            sweep_length=sweep_length
        )

        beta_results = {}
        for name, variable in tracker.history.items():
            beta_results[name] = variable.numpy()
        results[beta] = beta_results

        # Calculate Magnetic Susceptibility using the final state
        susceptibility = MagneticSusceptibility(
            ising_system).compute().numpy()  # type: ignore
        print(f"Susceptibility for beta={beta:.4f}: {susceptibility}")
        susceptibility_results.append((beta, susceptibility, 0.0))

    print("Plotting results...")

    num_plots = len(betas)
    cols = int(ceil(sqrt(num_plots)))
    rows = int(ceil(num_plots / cols))

    # Evolution Plots
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_flat = axes.flatten()

    # Color cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, beta in enumerate(betas):
        ax = axes_flat[i]
        mag_data = results[beta]['Magnetization']
        steps = np.arange(mag_data.shape[0]) * granularity

        # Plot each replica with a different color (cycling)
        for r in range(min(5, lattice_replicas)):  # Plot first 5 replicas to avoid clutter
            ax.plot(steps, mag_data[:, r], color=colors[r %
                    len(colors)], alpha=0.4, linewidth=1)

        mean_mag = np.mean(mag_data, axis=1)
        ax.plot(steps, mean_mag, color='black', alpha=1.0,
                linewidth=2, linestyle='--', label='Mean')

        ax.set_title(f'Beta={beta:.3f}')
        ax.set_xlabel('Step')
        if i % cols == 0:
            ax.set_ylabel('Magnetization')

    # Hide unused subplots
    for j in range(num_plots + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.savefig('examples/ising_evolution.png')
    print("Saved plot to examples/ising_evolution.png")

    # Susceptibility Plot
    plt.figure(figsize=(10, 6))
    betas_np, chi_means, chi_stds = zip(*susceptibility_results)

    plt.errorbar(betas_np, chi_means, yerr=chi_stds, fmt='-o',
                 color='purple', ecolor='gray', capsize=3)
    plt.axvline(x=0.44068, color='green', linestyle='--',
                label='Critical Beta (Onsager)')

    plt.title('Magnetic Susceptibility vs Beta')
    plt.xlabel('Inverse Temperature (Beta)')
    plt.ylabel('Susceptibility ($\chi$)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('examples/ising_susceptibility.png')
    print("Saved plot to examples/ising_susceptibility.png")


if __name__ == "__main__":
    run_simulation()
