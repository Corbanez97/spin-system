import os
from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.dynamics.tracker import Tracker
from spin_engine.dynamics import MetropolisHastings
from spin_engine.models import IsingSystem
from typing import List, Tuple, cast
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.config.optimizer.set_jit(True)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class FastMetropolis(MetropolisHastings):
    """
    Optimized Metropolis-Hastings using vectorized random sampling.
    Significantly faster for large number of replicas as it avoids 
    creating N_replicas independent shuffle operations in the graph.
    """

    def flip_spins(self, num_flips: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Generate random indices for each replica: (replicas, num_flips)
        replica_indices = tf.random.uniform(
            shape=(self.system.lattice_replicas, num_flips),
            minval=0,
            maxval=tf.cast(self.system.number_spins, tf.int32),
            dtype=tf.int32
        )

        row_indices = tf.range(
            self.system.lattice_replicas, dtype=tf.int32)[:, None]
        row_indices = tf.repeat(row_indices, num_flips, axis=1)

        scatter_indices = tf.stack([row_indices, replica_indices], axis=-1)
        scatter_indices = tf.reshape(scatter_indices, (-1, 2))

        spin_flat = tf.reshape(self.system.spin_state,
                               (self.system.lattice_replicas, -1))

        # Flip selected spins
        updates = tf.reshape(
            -tf.gather_nd(spin_flat, scatter_indices),
            [-1]
        )

        updated = tf.tensor_scatter_nd_update(
            spin_flat, scatter_indices, updates)
        updated = tf.reshape(updated, self.system.spin_state.shape)

        updated_energy = self.system.compute_energy(updated)

        return updated, updated_energy


def generate_betas(num_betas: int = 12, critical_beta: float = 0.44068) -> List[float]:
    """
    Generates a list of betas concentrated around the critical temperature.
    """
    dense_range = 0.1
    num_dense = int(0.6 * num_betas)
    num_sparse = num_betas - num_dense

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


def reshape_to_super_replicas(spin_state: tf.Tensor,
                              num_super: int,
                              grid_size: int) -> tf.Tensor:
    """
    Reshapes a collection of small replicas into larger 'super-replicas' by tiling them.

    Args:
        spin_state: Tensor of shape (total_replicas, sub_L, sub_L)
        num_super: Number of super-replicas to form.
        grid_size: Number of small replicas along one dimension of the super-replica (k).

    Returns:
        Tensor of shape (num_super, sub_L * k, sub_L * k)
    """
    total_replicas, sub_L_y, sub_L_x = spin_state.shape

    # ensure total_replicas matches expected size
    assert total_replicas == num_super * grid_size * grid_size

    # The grid is (grid_size, grid_size) blocks.
    # We first reshape to separate super-replicas and the grid blocks.
    # Shape: (num_super, grid_size, grid_size, sub_L, sub_L)
    reshaped = tf.reshape(
        spin_state, (num_super, grid_size, grid_size, sub_L_y, sub_L_x))

    # We want to form images where rows are composed of (grid_row, pixel_row)
    # and cols are composed of (grid_col, pixel_col).
    # Expected dim order for reshape: (num_super, grid_row, pixel_row, grid_col, pixel_col)
    # Current indices: 0, 1, 2, 3, 4
    # Target indices: 0, 1, 3, 2, 4
    transposed = tf.transpose(reshaped, perm=[0, 1, 3, 2, 4])

    # Final reshape to merge dimensions
    # new_H = grid_size * sub_L_y
    # new_W = grid_size * sub_L_x
    new_H = grid_size * sub_L_y
    new_W = grid_size * sub_L_x

    final_shape = (num_super, new_H, new_W)
    return tf.reshape(transposed, final_shape)


def compute_susceptibility(spin_state: tf.Tensor) -> float:
    """
    Computes magnetic susceptibility as variance of mean magnetization across replicas.
    """
    replicas = spin_state.shape[0]
    flat_state = tf.reshape(spin_state, (replicas, -1))
    m_per_replica = tf.reduce_mean(flat_state, axis=1)
    return float(tf.math.reduce_variance(m_per_replica).numpy())


def run_simulation():
    # Simulation Parameters
    sub_L = 16
    target_L = 64
    grid_size = target_L // sub_L  # 4
    num_super_replicas = 32

    replicas_per_super = grid_size * grid_size  # 16
    total_replicas = num_super_replicas * replicas_per_super  # 512

    print(f"Configuration:")
    print(f"  Sub-lattice size: {sub_L}x{sub_L}")
    print(f"  Target super-lattice size: {target_L}x{target_L}")
    print(f"  Grid tiling: {grid_size}x{grid_size}")
    print(f"  Total elementary replicas: {total_replicas}")
    print(f"  Total super-replicas formed: {num_super_replicas}")

    # Generate Interaction Matrix (for sub-lattice)
    interaction_matrix = PeriodicNearestNeighborInteraction().generate(2, sub_L)

    betas = generate_betas(24)
    print(f"Betas: {betas}")

    results_small = []
    results_stitched = []

    # Store one representative super-replica final state for each beta for visualization
    # Shape: (num_betas, target_L, target_L)
    slider_snapshots = []

    # Initialize System and Simulation ONCE to avoid retracing
    system = IsingSystem(
        lattice_dim=2,
        lattice_length=sub_L,
        lattice_replicas=total_replicas,
        interaction_matrix=interaction_matrix,
        initial_magnetization=0.5
    )
    simulation = FastMetropolis(system)
    tracker = Tracker([], granularity=5000)

    for beta in betas:
        print(f"Running Beta={beta:.4f}...", end="", flush=True)

        # Reset state for new beta
        new_init_state = system.initialize_state()
        system.update_state(new_init_state)
        # We must manually update current energy in simulation because we changed state externally
        simulation.current_energy.assign(system.compute_energy())

        # Run simulation (no tracking required, we only need final state)
        # Enough steps to equilibrate
        simulation.sweep(
            tracker=tracker,
            beta=tf.constant(beta, dtype=tf.float32),
            num_disturbances=tf.constant(1),
            sweep_length=2000
        )

        final_state = system.spin_state  # (total_replicas, sub_L, sub_L)

        # 1. Compute Susceptibility for SMALL replicas
        chi_small = compute_susceptibility(final_state)
        results_small.append(chi_small)

        # 2. Reshape to SUPER replicas
        super_state = reshape_to_super_replicas(
            final_state, num_super_replicas, grid_size)

        # 3. Compute Susceptibility for SUPER replicas
        chi_stitched = compute_susceptibility(super_state)
        results_stitched.append(chi_stitched)

        # 4. Save first super-replica for visualization
        slider_snapshots.append(super_state[0].numpy())  # (target_L, target_L)

        print(
            f" Done. Chi_Small={chi_small:.4f}, Chi_Stitched={chi_stitched:.4f}")

    # --- Plotting ---

    # 1. Static Susceptibility Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(betas, results_small, 'o--',
             label=f'Small Replicas ({sub_L}x{sub_L})', alpha=0.7)
    plt.plot(betas, results_stitched, 's-',
             label=f'Stitched Replicas ({target_L}x{target_L})', linewidth=2)

    plt.axvline(x=0.44068, color='green', linestyle=':',
                label='Critical Beta (Onsager)')
    plt.title('Magnetic Susceptibility: Independent vs Stitched Replicas')
    plt.xlabel('Inverse Temperature (Beta)')
    plt.ylabel(r'Susceptibility ($\chi$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('examples/ising_reshaped_susceptibility.png')
    print("Saved susceptibility plot to examples/ising_reshaped_susceptibility.png")

    # 2. Interactive Slider Plot (Plotly)
    snapshots = np.array(slider_snapshots)  # (num_betas, target_L, target_L)

    fig = go.Figure()

    # Add one trace (Heatmap)
    # Initial state (first beta, usually high temp / low beta)
    fig.add_trace(go.Heatmap(
        z=snapshots[0],
        colorscale='RdBu',
        zmin=-1, zmax=1,
        showscale=False
    ))

    # Create frames for slider
    frames = []
    for i, beta in enumerate(betas):
        frames.append(go.Frame(
            data=[go.Heatmap(z=snapshots[i])],
            name=f'beta_{beta:.4f}'
        ))

    fig.frames = frames

    # Slider configuration
    steps = []
    for i, beta in enumerate(betas):
        step = dict(
            method="animate",
            args=[[f'beta_{beta:.4f}'],
                  {"mode": "immediate",
                   "frame": {"duration": 300, "redraw": True},
                   "transition": {"duration": 0}}],
            label=f"{beta:.3f}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Beta: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title=f"Stitched Lattice State ({target_L}x{target_L}) vs Beta",
        width=700,
        height=700,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, scaleanchor="x", scaleratio=1),
    )

    fig.write_html("examples/ising_reshaped_slider.html")
    print("Saved slider plot to examples/ising_reshaped_slider.html")


if __name__ == "__main__":
    run_simulation()
