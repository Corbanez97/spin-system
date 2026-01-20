from spin_engine.models.traveling_salesman import TravelingSalesmanSystem
import sys
import os
import pytest
import tensorflow as tf
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../src')))


class TestTSPModel:

    @pytest.fixture
    def tsp_system(self):
        """
        Creates a basic TSP system for testing.
        3 Nodes (L=3), 2 Replicas.
        """
        L = 3
        replicas = 2
        # Simple distance matrix (symmetric, zero diagonal)
        # 0--1: 1.0
        # 1--2: 2.0
        # 0--2: 3.0
        cost_matrix = tf.constant([
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0]
        ], dtype=tf.float32)

        return TravelingSalesmanSystem(
            cost_matrix=cost_matrix,
            lattice_replicas=replicas
        )

    def test_initialization_constraints(self, tsp_system):
        """
        Test that the initialized state satisfies TSP constraints:
        1. Values are strictly -1 or 1.
        2. When mapped to 0/1 (binary), row sums = 1 (each city visited once).
        3. Column sums = 1 (one city per time step).
        """
        # Initialize
        spin_state = tsp_system.initialize_state()

        # Check shape: (R, L, L)
        assert spin_state.shape == (2, 3, 3)

        # Check values are -1 or 1
        # We can check by abs(x) == 1
        tf.debugging.assert_near(
            tf.abs(spin_state), tf.ones_like(spin_state), atol=1e-5)

        # Convert to binary (0, 1)
        binary_state = (spin_state + 1.0) / 2.0

        # Check row sums (Sum over columns, axis 2) -> Should be 1
        row_sums = tf.reduce_sum(binary_state, axis=2)
        tf.debugging.assert_near(row_sums, tf.ones_like(row_sums), atol=1e-5)

        # Check col sums (Sum over rows, axis 1) -> Should be 1
        col_sums = tf.reduce_sum(binary_state, axis=1)
        tf.debugging.assert_near(col_sums, tf.ones_like(col_sums), atol=1e-5)

    def test_matrix_compatibility(self):
        """
        Test that the system correctly infers lattice length from cost matrix
        and validates compatibility.
        """
        L = 4
        cost_matrix = tf.random.uniform((L, L))

        system = TravelingSalesmanSystem(
            cost_matrix=cost_matrix,
            lattice_replicas=5
        )

        # Check inferred length
        assert system.lattice_length == L

        # Check init shape
        state = system.initialize_state()
        assert state.shape == (5, L, L)
