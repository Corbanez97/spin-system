import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import cast

from spin_engine.models import SherringtonKirkpatrickSystem
from spin_engine.dynamics import MetropolisHastings
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.correlations import OverlapDistribution, ParisiOverlapParameter
from spin_engine.measurements.scalars import SpinGlassOrderParameter


def run_simulation():
    lattice_length = 64
    lattice_dim = 1
    # 64 replicas gives 64*63/2 = 2016 pairs for smooth P(q)
    lattice_replicas = 64
    J = 1.0

    granularity = 100
    num_flips = cast(tf.Tensor, tf.constant(1))
    
    N = lattice_length ** lattice_dim
    sweeps = 2000
    sweep_length = sweeps * N

    betas = [0.2, 0.5, 0.8, 0.9, 1.0, 1.1, 1.5, 2.0]
    
    results = {}
    
    print(f"Starting Phase Diagram Simulation with {len(betas)} betas: {betas}")
    print(f"  SK: L={lattice_length}, D={lattice_dim}, N={N}, "
          f"replicas={lattice_replicas}, sweeps={sweeps} (total steps={sweep_length})")

    # Initialize ONCE outside the loop to enable Simulated Annealing (hot to cold)
    sk_system = SherringtonKirkpatrickSystem(
        lattice_length=lattice_length,
        lattice_dim=lattice_dim,
        lattice_replicas=lattice_replicas,
        J=J,
        initial_magnetization=0.0 # Hot state (T=infinity)
    )
    simulation = MetropolisHastings(sk_system)
    
    q_ea_measurement = SpinGlassOrderParameter(sk_system)
    
    tracker = Tracker(measurements=[
        OverlapDistribution(sk_system),
        ParisiOverlapParameter(sk_system),
        q_ea_measurement
    ], granularity=granularity)

    for beta in betas:
        print(f"Running for beta={beta:.4f}")
        
        # Reset stateful measurements between beta sweeps
        q_ea_measurement.reset()
        
        simulation.sweep(
            tracker=tracker,
            beta=tf.constant(beta, dtype=tf.float32),
            num_disturbances=num_flips,
            sweep_length=sweep_length
        )
        
        beta_results = {}
        for name, variable in tracker.history.items():
            beta_results[name] = variable.numpy()
            
        results[beta] = beta_results

    print("Plotting results...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot q_EA vs T ---
    ax1 = axes[0]
    temps = [1.0/b for b in betas]
    
    q_ea_means = []
    q_ea_stds = []
    
    for beta in betas:
        q_ea_history = results[beta]['SpinGlassOrderParameter']
        # Average over the second half of the sweep
        half_idx = len(q_ea_history) // 2
        q_ea_means.append(np.mean(q_ea_history[half_idx:]))
        q_ea_stds.append(np.std(q_ea_history[half_idx:]))
        
    ax1.errorbar(temps, q_ea_means, yerr=q_ea_stds, fmt='o-', color='crimson', markersize=6, capsize=4)
    ax1.axvline(x=1.0, color='black', linestyle='--', label=r'$T_c = 1.0$')
    ax1.set_xlabel('Temperature $T = 1/\\beta$')
    ax1.set_ylabel('Edwards-Anderson Parameter $q_{EA}$')
    ax1.set_title('Spin Glass Phase Transition')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Plot P(q) ---
    ax2 = axes[1]
    
    selected_betas = [0.5, 0.8, 1.0, 1.5, 2.0]
    palette = sns.color_palette("coolwarm", len(selected_betas))
    
    for i, beta in enumerate(selected_betas):
        if beta in results:
            pq_history = results[beta]['OverlapDistribution']
            half_idx = len(pq_history) // 2
            # Flatten the last half of the measurements to get the equilibrium P(q)
            q_values = pq_history[half_idx:].flatten()
            
            sns.kdeplot(q_values, ax=ax2, color=palette[i], fill=True, alpha=0.2, label=f'T={1.0/beta:.2f}')
            
    ax2.set_xlabel('Overlap $q$')
    ax2.set_ylabel('$P(q)$')
    ax2.set_title('Overlap Distribution $P(q)$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if not os.path.exists('examples/images'):
        os.makedirs('examples/images')
        
    save_path = 'examples/images/spin_glass_phase_diagram.png'
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    run_simulation()
