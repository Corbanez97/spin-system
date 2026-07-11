"""
Edwards-Anderson (EA) Spin Glass — Observables and Parisi Distribution Plot.

This script simulates the 3D EA model across a range of temperatures/betas
for multiple lattice sizes, calculates the Specific Heat, Overlap, Overlap
Susceptibility, and the Parisi Overlap Distribution, and saves a 2x2 grid plot
using premium publication-quality style parameters.

It stores simulation results incrementally per lattice size L into JSON files in
examples/data/ to prevent data loss, enable resuming runs, and group datasets.
"""

import os
import glob
import json
import argparse
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
    num_betas: int = 25,
    critical_beta: float = 0.892,
    min_beta: float = 0.2,
    max_beta: float = 2.5,
) -> List[float]:
    """Generates a list of betas concentrated around the critical temperature (Tc ~ 1.124 => beta_c ~ 0.892)."""
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


def run_simulation(is_test: bool = False, force_run: bool = False):
    # Simulation Parameters
    if is_test:
        L_list = [4]
        lattice_replicas = 16
        betas = generate_betas(3, min_beta=1/3, max_beta=2.5)
        sweeps = 200
    else:
        L_list = [6, 8, 10, 12, 16]
        lattice_replicas = 512
        betas = generate_betas(25, min_beta=1/3, max_beta=2.5)
        sweeps = 1000
        
    J = 1.0
    coupling_seed = 42
    num_flips = cast(tf.Tensor, tf.constant(1))
    
    data_dir = 'examples/data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Starting Edwards-Anderson Spin Glass Observables Simulation (3D)")
    print(f"Lattice sizes: {L_list}")
    print(f"Number of betas: {len(betas)}")
    print(f"Replicas: {lattice_replicas}")
    print(f"Sweeps per temperature: {sweeps}")
    
    for L in L_list:
        N = L ** 3
        sweep_length = sweeps * N
        # Scale granularity to always record ~200 steps to prevent OOM errors
        granularity = max(1, sweep_length // 200)

        cache_file = os.path.join(data_dir, f"ea_observables_L{L}.json")
        
        # Caching/Resuming Check
        if os.path.exists(cache_file) and not force_run:
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Verify that parameters match to prevent loading wrong configurations
                betas_match = len(cached_data.get('betas', [])) == len(betas) and np.allclose(cached_data.get('betas', []), betas)
                params_match = (
                    cached_data.get('lattice_replicas') == lattice_replicas and
                    cached_data.get('sweeps') == sweeps and
                    cached_data.get('granularity') == granularity and
                    betas_match
                )
                
                if params_match:
                    print(f"\n--- Found cached data for L={L} ({cache_file}). Skipping simulation. ---")
                    continue
                else:
                    print(f"\n--- Parameter mismatch in cache for L={L}. Re-running simulation. ---")
            except Exception as e:
                print(f"\n--- Error reading cache for L={L} ({e}). Re-running simulation. ---")
        
        # Build quenched coupling matrix (same for all betas)
        nn_mask = PeriodicNearestNeighborInteraction().generate(3, L)
        random_J = BinaryRandomInteraction(J=J, seed=coupling_seed).generate(3, L)
        interaction_matrix = nn_mask * random_J
        
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
        
        q2_list = []
        chi_sg_list = []
        cv_list = []
        binder_list = []
        parisi_hists = []
        q_bins = None
        
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
            q4_avg = np.mean(pq_flat**4)
            chi_sg = beta * N * q2_avg
            binder = 0.5 * (3.0 - q4_avg / (q2_avg**2)) if q2_avg > 0 else 0.0
            
            cv_list.append(float(cv))
            q2_list.append(float(q2_avg))
            chi_sg_list.append(float(chi_sg))
            binder_list.append(float(binder))
            
            hist, bin_edges = np.histogram(pq_flat, bins=100, range=(-1.05, 1.05), density=True)
            if q_bins is None:
                q_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
            parisi_hists.append(hist.tolist())
            
            print(f"  beta={beta:.4f} | T={1/beta:.3f} | <q^2>={q2_avg:.4f} | chi_SG={chi_sg:.4f} | cv={cv:.4f} | U={binder:.4f}")
            
        # Write results for this size L to its individual JSON file
        size_data = {
            'L': L,
            'lattice_replicas': lattice_replicas,
            'sweeps': sweeps,
            'granularity': granularity,
            'betas': betas,
            'q2': q2_list,
            'chi_sg': chi_sg_list,
            'cv': cv_list,
            'binder': binder_list,
            'parisi_hists': parisi_hists,
            'q_bins': q_bins.tolist() if q_bins is not None else []
        }
        
        with open(cache_file, 'w') as f:
            json.dump(size_data, f)
        print(f"Saved L={L} results to {cache_file}")
        
    print("\nSimulation phase complete. Compiling all available data for plotting...")
    
    # Compile results for all matching JSON files in examples/data/
    file_pattern = os.path.join(data_dir, "ea_observables_L*.json")
    all_files = glob.glob(file_pattern)
    
    plot_data: Dict[str, Any] = {
        'L_list': [],
        'betas': betas,
        'data': {},
        'parisi': {},
        'q_bins': []
    }
    
    for fp in all_files:
        try:
            with open(fp, 'r') as f:
                d = json.load(f)
            
            # Verify if this file's parameters match the current configuration
            betas_match = len(d.get('betas', [])) == len(betas) and np.allclose(d.get('betas', []), betas)
            L_val = d.get('L', 0)
            expected_granularity = max(1, (sweeps * (L_val**3)) // 200)
            params_match = (
                d.get('lattice_replicas') == lattice_replicas and
                d.get('sweeps') == sweeps and
                d.get('granularity') == expected_granularity and
                betas_match
            )
            
            if params_match:
                L_str = str(L_val)
                plot_data['L_list'].append(L_val)
                plot_data['data'][L_str] = {
                    'q2': d['q2'],
                    'chi_sg': d['chi_sg'],
                    'cv': d['cv'],
                    'binder': d.get('binder', [])
                }
                plot_data['parisi'][L_str] = d.get('parisi_hists', [])
                if not plot_data['q_bins'] and 'q_bins' in d:
                    plot_data['q_bins'] = d['q_bins']
        except Exception as e:
            print(f"Warning: Failed to load file {fp} for plotting: {e}")
            
    # Sort the lattice sizes for plotting
    plot_data['L_list'] = sorted(list(set(plot_data['L_list'])))
    
    if not plot_data['L_list']:
        print("Error: No valid datasets found matching current parameters.")
        return
        
    # Save the compiled results summary to JSON for convenience
    os.makedirs('examples/output', exist_ok=True)
    summary_file = 'examples/output/ea_observables_results.json'
    with open(summary_file, 'w') as f:
        json.dump(plot_data, f)
    print(f"Saved compiled summary JSON to {summary_file}")
    
    print(f"Plotting results for lattice sizes: {plot_data['L_list']}")
    plot_results(plot_data)


def plot_results(results_dict: Dict[str, Any]):
    L_list = results_dict['L_list']
    betas = np.array(results_dict['betas'])
    temps = 1.0 / betas
    data = results_dict['data']
    parisi = results_dict['parisi']
    q_bins = np.array(results_dict['q_bins'])
    
    Tc_exact = 1.124  # Expected critical temperature in 3D
    
    # Set premium publication-quality style parameters
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 15,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(L_list))]
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    
    for i, L in enumerate(L_list):
        L_str = str(L)
        color = colors[i]
        
        # 1. Overlap vs T
        axs[0].plot(temps, data[L_str]['q2'], 'o-', color=color, markersize=4, linewidth=1.5, label=f'L={L}')
        
        # 2. Overlap Susceptibility vs T
        axs[1].plot(temps, data[L_str]['chi_sg'], 's-', color=color, markersize=4, linewidth=1.5, label=f'L={L}')
        
        # 3. Binder Cumulant vs T
        axs[2].plot(temps, data[L_str]['binder'], 'd-', color=color, markersize=4, linewidth=1.5, label=f'L={L}')
        
        # 4. Specific Heat vs T
        axs[3].plot(temps, data[L_str]['cv'], '^-', color=color, markersize=4, linewidth=1.5, label=f'L={L}')
        
        # 5. Parisi Distribution at lowest temperature
        if len(parisi[L_str]) > 0:
            lowest_temp_hist = parisi[L_str][-1]  # Highest beta is last
            axs[4].plot(q_bins, lowest_temp_hist, '-', color=color, linewidth=2, label=f'L={L}')
            axs[4].fill_between(q_bins, 0, lowest_temp_hist, color=color, alpha=0.15)
        
    # 6. Parisi Distribution Heatmap vs T (for largest L)
    largest_L = str(L_list[-1])
    if len(parisi[largest_L]) > 0:
        parisi_matrix = np.array(parisi[largest_L]).T  # shape (len(q_bins), len(betas))
        sort_idx = np.argsort(temps)
        sorted_temps = temps[sort_idx]
        sorted_matrix = parisi_matrix[:, sort_idx]
        
        T_mesh, Q_mesh = np.meshgrid(sorted_temps, q_bins)
        c = axs[5].pcolormesh(T_mesh, Q_mesh, sorted_matrix, cmap='magma', shading='auto')
        fig.colorbar(c, ax=axs[5], label=r'$P(q)$ Density')
        axs[5].axvline(Tc_exact, color='cyan', linestyle=':', linewidth=2, alpha=0.8, label=r'$T_c \approx 1.124$')
    
    # Add labels and format plots
    for idx in range(4):
        ax = axs[idx]
        ax.axvline(Tc_exact, color='crimson', linestyle=':', linewidth=1.5, alpha=0.8, label=r'$T_c \approx 1.124$')
        ax.set_xlabel(r'$T = 1/\beta$')
        ax.grid(True, alpha=0.3)
        
    axs[0].set_ylabel(r'$\langle q^2 \rangle$')
    axs[0].set_title('Spin Glass Order Parameter')
    
    axs[1].set_ylabel(r'$\chi_{\rm SG} = \beta N \langle q^2 \rangle$')
    axs[1].set_title('Overlap Susceptibility')
    
    axs[2].set_ylabel(r'$U = \frac{1}{2} \left(3 - \frac{\langle q^4 \rangle}{\langle q^2 \rangle^2}\right)$')
    axs[2].set_title('Binder Cumulant')
    
    axs[3].set_ylabel(r'$C_v$')
    axs[3].set_title('Specific Heat')
    
    axs[4].set_xlabel(r'$q$')
    axs[4].set_ylabel(r'$P(q)$')
    axs[4].set_title(f'Parisi Overlap Distribution at $T = {1.0/betas[-1]:.2f}$')
    axs[4].grid(True, alpha=0.3)
    axs[4].axvline(0, color='black', linewidth=0.8, alpha=0.5)
    
    axs[5].set_xlabel(r'$T = 1/\beta$')
    axs[5].set_ylabel(r'$q$')
    axs[5].set_title(f'Evolution of P(q) vs T (L={largest_L})')
    
    # Single unified legend to avoid crowding
    handles, labels = axs[0].get_legend_handles_labels()
    unique_labels = {}
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels[l] = h
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=len(unique_labels), borderaxespad=0.)
    
    plt.suptitle('3D Edwards-Anderson Spin Glass Observables', y=1.02, weight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    os.makedirs('examples/images', exist_ok=True)
    plt.savefig('examples/images/ea_observables.png', dpi=300, bbox_inches='tight')
    print("Saved premium observables plot to examples/images/ea_observables.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate 3D Edwards-Anderson model.")
    parser.add_argument("--test", action="store_true", help="Run a quick test simulation.")
    parser.add_argument("--force", action="store_true", help="Force re-running simulations even if cache exists.")
    args = parser.parse_args()
    
    run_simulation(is_test=args.test, force_run=args.force)
