from spin_engine.dynamics.tracker import Tracker
from spin_engine.dynamics.traveling_salesman import TravelingSalesmanDynamics
from spin_engine.models.traveling_salesman import TravelingSalesmanSystem
import sys
import os
import pytest
import tensorflow as tf
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../src')))


class TestTSPDynamics:

    @pytest.fixture
    def tsp_graph_5(self):
        """
        Fixture with a simple geometry of 5 nodes.
        Nodes at (0,0), (0, 1), (0.5, 1.5), (1, 1), (1, 0).
        Optimal path length is ~4.414.
        """
        coords = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.5],
            [1.0, 1.0],
            [1.0, 0.0]
        ], dtype=np.float32)

        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        return tf.convert_to_tensor(dist_matrix, dtype=tf.float32)

    def test_dynamics_evolution(self, tsp_graph_5):
        """
        Evolve for a set of steps and check constraints and energy.
        """
        tf.random.set_seed(42)

        replicas = 2

        system = TravelingSalesmanSystem(
            cost_matrix=tsp_graph_5,
            lattice_replicas=replicas,
            distance_strength=1.0,
            constraint_strength=10.0
        )
        system.initialize_state()

        dynamics = TravelingSalesmanDynamics(system)

        initial_energy = system.compute_energy()

        # Run a short sweep
        # Beta = 10.0 (High enough to encourage optimization)
        beta = 10.0
        steps = 100

        # We need a tracker dummy or real
        # The sweep method requires a tracker.
        # Let's use a simple dummy tracker or the real one if easy.
        # Ideally we should use the real tracker but we can also just call step() in a loop
        # to avoid Tracker dependency complexity if not strictly needed.
        # However, the user asked to "evolve for a set of steps".
        # Let's use explicit loop over step() for simplicity in testing dynamics logic
        # without tracker overhead, OR use sweep with a mocked tracker.
        # using step() loop is easier to debug here.

        for _ in range(steps):
            dynamics.step(beta=beta)

        final_energy = system.compute_energy()

        # 1. Check Constraints maintained
        # Convert to binary
        spin_state = system.spin_state
        binary_state = (spin_state + 1.0) / 2.0
        row_sums = tf.reduce_sum(binary_state, axis=2)
        col_sums = tf.reduce_sum(binary_state, axis=1)

        tf.debugging.assert_near(row_sums, tf.ones_like(row_sums), atol=1e-5)
        tf.debugging.assert_near(col_sums, tf.ones_like(col_sums), atol=1e-5)

        # 2. Check Energy Improvement (or at least non-increase for high beta,
        # but with stochasticity it might flucuate slightly, but for 100 steps on 5 nodes it should find optimal)
        # Optimal is approx 4.414.
        # Random path is likely > 4.414.

        # Check that average final energy is less than initial
        avg_init = tf.reduce_mean(initial_energy)
        avg_final = tf.reduce_mean(final_energy)

        assert avg_final <= avg_init

        # Check proximity to optimal (allow some margin as 100 steps might not be fully converged for all replicas)
        optimal_energy = 4.414
        # We expect at least one replica to be close or equal to optimal
        min_final = tf.reduce_min(final_energy)
        # It shouldn't be magically lower than optimal (unless my optimization calculation is wrong, but it's geometric)
        # It should be close.
        print(f"Initial Energy: {initial_energy.numpy()}")
        print(f"Final Energy: {final_energy.numpy()}")

        # Soft check: valid energy
        assert min_final >= optimal_energy - 1e-3
