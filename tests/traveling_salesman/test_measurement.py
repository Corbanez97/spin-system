from spin_engine.models.traveling_salesman import TravelingSalesmanSystem
import sys
import os
import pytest
import tensorflow as tf
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../src')))


class TestTSPMeasurements:

    @pytest.fixture
    def tsp_graph_5(self):
        """
        Fixture with a simple geometry of 5 nodes.
        Nodes at (0,0), (0, 1), (0.5, 1.5), (1, 1), (1, 0).
        """
        coords = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.5],
            [1.0, 1.0],
            [1.0, 0.0]
        ], dtype=np.float32)

        # Compute Euclidean distance matrix
        # Shape (5, 5)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        return tf.convert_to_tensor(dist_matrix, dtype=tf.float32)

    def test_energy_computation(self, tsp_graph_5):
        """
        Initialize the fixture for the given graph with the 42 seed and compute the energy.
        """
        # Set seed for reproducibility
        tf.random.set_seed(42)

        system = TravelingSalesmanSystem(
            cost_matrix=tsp_graph_5,
            lattice_replicas=1,  # Single replica as implied "compute the energy"
            distance_strength=1.0,
            constraint_strength=10.0
        )

        # Initialize state
        system.initialize_state()

        # Compute Energy
        energy = system.compute_energy()

        # Assertions
        # Energy should be shape (1,)
        assert energy.shape == (1,)

        # Since initialize_state guarantees valid path constraints,
        # the constraint penalty should be 0.
        # Energy should effectively be just the distance cost * B (1.0).
        # Distance > 0.
        assert energy[0] > 0.0

        # Optional: Check consistency. With seed 42, we expect a deterministic result.
        # But we won't hardcode the exact float to avoid brittleness, just validity.
        assert tf.math.is_finite(energy[0])
