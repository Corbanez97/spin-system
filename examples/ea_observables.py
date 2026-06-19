"""
Edwards-Anderson (EA) Spin Glass — Observables and Parisi Distribution Plot.

This script simulates the 3D EA model across a range of temperatures/betas
for multiple lattice sizes, calculates the Specific Heat, Overlap, Overlap
Susceptibility, and the Parisi Overlap Distribution, and saves a 2x2 grid plot
similar to `ising_observables.png`.
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import cast, List, Dict, Any

from spin_engine.models import EdwardsAndersonSystem
from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.interactions.standard import BinaryRandomInteraction
from spin_engine.dynamics import MetropolisHastings
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.scalars import Energy
from spin_engine.measurements.correlations import OverlapDistribution


def generate_betas(
    num_betas: int = 20,
    critical_beta: float = 0.909,
    min_beta: float = 0.2,
    max_beta: float = 2.5,
) -> List[float]:
    """Generates a list of betas concentrated around the critical temperature (Tc ~ 1.1 => beta_c ~ 0.909)."""
    dense_range = 0.2
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


def run_simulation():
    # Simulation Parameters
    L_list = [4, 6]
    lattice_replicas = 64
    betas = generate_betas(5)
    J = 1.0
    coupling_seed = 42
    granularity = 100
    
    num_flips = cast(tf.Tensor, tf.constant(1))
    
    results_dict: Dict[str, Any] = {
        'L_list': L_list,
        'betas': betas,
        'data': {},
        'parisi': {} # lowest temp P(q) distribution for each L
    }
    
    print("Starting Edwards-Anderson Spin Glass Observables Simulation (3D)")
    print(f"Lattice sizes: {L_list}")
    print(f"Number of betas: {len(betas)}")
    
    for L in L_list:
        results_dict['data'][str(L)] = {
            'q2': [], 'chi_sg': [], 'cv': []
        }
        
        N = L ** 3
        
        # Build quenched coupling matrix (same for all betas)
        nn_mask = PeriodicNearestNeighborInteraction().generate(3, L)
        random_J = BinaryRandomInteraction(J=J, seed=coupling_seed).generate(3, L)
        interaction_matrix = nn_mask * random_J
        
        # Define sweep length dynamically based on system size L to ensure proper equilibration
        if L == 4:
            sweeps = 4000
        elif L == 6:
            sweeps = 3000
        else: # L == 8
            sweeps = 2000
            
        sweep_length = sweeps * N
        burn_in_steps = int((sweep_length / granularity) * 0.5)  # Discard first 50% for equilibration
        
        print(f"\n--- Running for Lattice Size L={L} (sweeps={sweeps}, total steps={sweep_length}) ---")
        
        # Initialize system outside the loop for Simulated Annealing (hot to cold)
        system = EdwardsAndersonSystem(
            lattice_length=L,
            lattice_dim=3,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix,
            initial_magnetization=0.0, # Hot state
        )
        simulation = MetropolisHastings(system)
        
        tracker = Tracker(
            measurements=[
                Energy(system),
                OverlapDistribution(system),
            ],
            granularity=granularity,
        )
        
        # Ascending betas = decreasing temperature (Annealing)
        for beta in betas:
            simulation.sweep(
                tracker=tracker,
                beta=tf.constant(beta, dtype=tf.float32),
                num_disturbances=num_flips,
                sweep_length=sweep_length,
            )
            
            # Extract data and discard burn-in
            E_hist = tracker.history['Energy'].numpy()[burn_in_steps:, :] # shape: (steps, replicas)
            pq_hist = tracker.history['OverlapDistribution'].numpy()[burn_in_steps:, :] # shape: (steps, num_pairs)
            
            # Normalize energy per spin
            e_hist = E_hist / N
            e_flat = e_hist.flatten()
            
            e_avg = np.mean(e_flat)
            e2_avg = np.mean(e_flat**2)
            
            # Compute observables
            cv = (beta**2) * N * (e2_avg - e_avg**2)
            
            pq_flat = pq_hist.flatten()
            q2_avg = np.mean(pq_flat**2)
            chi_sg = beta * N * q2_avg
            
            results_dict['data'][str(L)]['cv'].append(float(cv))
            results_dict['data'][str(L)]['q2'].append(float(q2_avg))
            results_dict['data'][str(L)]['chi_sg'].append(float(chi_sg))
            
            print(f"  beta={beta:.4f} | T={1/beta:.3f} | <q^2>={q2_avg:.4f} | chi_SG={chi_sg:.4f} | cv={cv:.4f}")
            
        # Store lowest temp (highest beta) overlap values for Parisi Distribution plot
        # the last beta checked is the highest beta (lowest temp)
        last_beta = betas[-1]
        results_dict['parisi'][str(L)] = pq_hist.flatten().tolist()
        
    # Save results to JSON for persistence
    os.makedirs('examples/output', exist_ok=True)
    with open('examples/output/ea_observables_results.json', 'w') as f:
        json.dump(results_dict, f)
        
    print("\nSimulation complete. Plotting results...")
    plot_results(results_dict)


def plot_results(results_dict: Dict[str, Any]):
    L_list = results_dict['L_list']
    betas = np.array(results_dict['betas'])
    temps = 1.0 / betas
    data = results_dict['data']
    parisi = results_dict['parisi']
    
    Tc_exact = 1.1  # Expected critical temperature in 3D
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, L in enumerate(L_list):
        L_str = str(L)
        c = colors[i % len(colors)]
        
        # 1. Overlap vs T
        axs[0].plot(temps, data[L_str]['q2'], 'o-', color=c, label=f'L={L}')
        
        # 2. Overlap Susceptibility vs T
        axs[1].plot(temps, data[L_str]['chi_sg'], 's-', color=c, label=f'L={L}')
        
        # 3. Specific Heat vs T
        axs[2].plot(temps, data[L_str]['cv'], '^-', color=c, label=f'L={L}')
        
        # 4. Parisi Distribution at lowest temperature
        sns.kdeplot(
            np.array(parisi[L_str]),
            ax=axs[3],
            color=c,
            fill=True,
            alpha=0.15,
            label=f'L={L}'
        )
        
    # Add labels and format plots
    for idx in range(3):
        ax = axs[idx]
        ax.axvline(Tc_exact, color='k', linestyle='--', alpha=0.5, label=r'$T_c \approx 1.1$')
        ax.set_xlabel(r'$T = 1/\beta$')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    axs[0].set_ylabel(r'$\langle q^2 \rangle$')
    axs[0].set_title('Spin Glass Order Parameter')
    
    axs[1].set_ylabel(r'$\chi_{\rm SG} = \beta N \langle q^2 \rangle$')
    axs[1].set_title('Overlap Susceptibility')
    
    axs[2].set_ylabel(r'$C_v$')
    axs[2].set_title('Specific Heat')
    
    # Format the Parisi Distribution subplot
    axs[3].set_xlabel(r'$q$')
    axs[3].set_ylabel(r'$P(q)$')
    axs[3].set_title(f'Parisi Overlap Distribution at $T = {1.0/betas[-1]:.2f}$')
    axs[3].grid(True, alpha=0.3)
    axs[3].legend()
    
    plt.tight_layout()
    os.makedirs('examples/images', exist_ok=True)
    plt.savefig('examples/images/ea_observables.png', dpi=150)
    print("Saved observables plot to examples/images/ea_observables.png")


if __name__ == "__main__":
    run_simulation()
