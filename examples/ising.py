import tensorflow as tf
import matplotlib.pyplot as plt
from typing import cast

from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.models import IsingSystem
from spin_engine.dynamics import MetropolisHastings
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.scalars import Energy, Magnetization


def run_simulation():
    lattice_dim = 2
    lattice_length = 32
    lattice_replicas = 32

    interaction_matrix = PeriodicNearestNeighborInteraction().generate(
        lattice_dim, lattice_length)

    granularity = 100
    num_flips = cast(tf.Tensor, tf.constant(1))

    betas = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    results = {}

    print("Starting simulation...")

    for beta in betas:
        print(f"Running for beta={beta}")

        ising_system = IsingSystem(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix,
            initial_magnetization=0.5
        )
        simulation = MetropolisHastings(ising_system)

        tracker = Tracker(measurements=[Energy(ising_system), Magnetization(
            ising_system)], granularity=granularity)

        simulation.sweep(
            tracker=tracker,
            beta=beta,
            num_disturbances=num_flips,
            sweep_length=10000
        )

        beta_results = {}
        for name, variable in tracker.history.items():
            beta_results[name] = variable.numpy()
        results[beta] = beta_results

    print("Plotting results...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, beta in enumerate(betas):
        ax_energy = axes[i, 0]
        energy_data = results[beta]['Energy']
        steps = range(0, energy_data.shape[0] * granularity, granularity)

        ax_energy.plot(steps, energy_data, color='blue',
                       alpha=0.3, linewidth=1)
        mean_energy = tf.reduce_mean(energy_data, axis=1)
        ax_energy.plot(steps, mean_energy, color='black',
                       alpha=1.0, linewidth=2, label='Mean')

        ax_energy.set_title(f'Energy Evolution (Beta={beta})')
        ax_energy.set_xlabel('Step')
        ax_energy.set_ylabel('Energy')

        ax_mag = axes[i, 1]
        mag_data = results[beta]['Magnetization']

        ax_mag.plot(steps, mag_data, color='red', alpha=0.3, linewidth=1)
        mean_mag = tf.reduce_mean(mag_data, axis=1)
        ax_mag.plot(steps, mean_mag, color='black',
                    alpha=1.0, linewidth=2, label='Mean')

        ax_mag.set_title(f'Magnetization Evolution (Beta={beta})')
        ax_mag.set_xlabel('Step')
        ax_mag.set_ylabel('Magnetization')

    plt.tight_layout()
    plt.savefig('ising_evolution.png')
    print("Saved plot to ising_evolution.png")


if __name__ == "__main__":
    run_simulation()
