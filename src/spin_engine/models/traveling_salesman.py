import tensorflow as tf
import numpy as np
from typing import Optional, Union, Callable, cast
from .base import BaseSpinSystem


class TravelingSalesmanSystem(BaseSpinSystem):
    def __init__(
        self,
        cost_matrix: Union[tf.Tensor, np.ndarray],
        lattice_replicas: int,
        constraint_strength: float = 10.0,
        distance_strength: float = 1.0,
        initial_spin_state: Optional[Union[tf.Tensor,
                                           Callable[[], tf.Tensor]]] = None,
    ):
        """
        Args:
            cost_matrix (LxL): Matrix defining distances between nodes.
            lattice_replicas: Number of parallel simulations.
            constraint_strength (A): Penalty for invalid paths (must be > max(cost)).
            distance_strength (B): Multiplier for the distance cost.
        """
        # Validate cost matrix shape
        self.cost_matrix = tf.convert_to_tensor(cost_matrix, dtype=tf.float32)
        L = self.cost_matrix.shape[0]
        L = cast(int, L)

        # Initialize Base.
        # Note: We treat the L x L grid as a 2D lattice of dimensions L, L
        super().__init__(
            lattice_dim=2,
            lattice_length=L,
            lattice_replicas=lattice_replicas,
            initial_spin_state=initial_spin_state
        )

        self.A = tf.constant(constraint_strength, dtype=tf.float32)
        self.B = tf.constant(distance_strength, dtype=tf.float32)

    def initialize_state(self) -> tf.Tensor:
        """
        Initializes random spins.
        """
        # Start with random -1/+1 spins
        full_shape = [self.lattice_replicas,
                      self.lattice_length, self.lattice_length]
        rand = tf.random.uniform(full_shape)
        return tf.where(rand > 0.5, 1.0, -1.0)

    # @tf.function
    def compute_energy(self, spin_state: Optional[tf.Variable | tf.Tensor] = None) -> tf.Tensor:
        """
        Computes H = H_constraints + H_distance
        """
        if spin_state is None:
            spin_state = self.spin_state

        # 1. Convert Ising spins {-1, 1} to Binary variables {0, 1}
        # x = (s + 1) / 2
        x = tf.divide(tf.add(spin_state, 1.0), 2.0)

        # --- H_A: Constraints ---
        # Sum over columns (axis 2: time steps) -> Should be 1
        row_sums = tf.reduce_sum(x, axis=2)
        # Sum over rows (axis 1: cities) -> Should be 1
        col_sums = tf.reduce_sum(x, axis=1)

        # Penalty: A * sum((1 - sum)^2)
        row_penalty = tf.reduce_sum(tf.square(1.0 - row_sums), axis=1)
        col_penalty = tf.reduce_sum(tf.square(1.0 - col_sums), axis=1)

        term_A = self.A * (row_penalty + col_penalty)

        # --- H_B: Distance Cost ---
        # We need sum_{u,v} W_{uv} * sum_j x_{u,j} * x_{v, j+1}

        # Shift x to get x_{v, j+1} (Rolling the time axis)
        # Using roll with shift -1 moves index 1 to 0, effectively getting "next step"
        x_next = tf.roll(x, shift=-1, axis=2)

        # Einstein summation to compute the cost efficiently:
        # r: replicas
        # u: source city (rows of x)
        # v: dest city (rows of x_next)
        # j: time step (cols of x)
        # W_{uv} * x_{r,u,j} * x_{r,v,j}

        # Step 1: Compute interaction between step j and j+1 for all replicas
        # Shape: (replicas, cities_u, cities_v)
        # Correlates "u at step j" with "v at step j+1"
        step_correlation = tf.matmul(x, x_next, transpose_b=True)

        # Step 2: Multiply by weight matrix and sum
        # element-wise multiply (broadcasting W over replicas) then sum matrix
        dist_cost = tf.reduce_sum(
            step_correlation * self.cost_matrix, axis=[1, 2])

        term_B = self.B * dist_cost

        return term_A + term_B
