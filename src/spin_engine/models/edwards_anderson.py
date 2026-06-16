import tensorflow as tf
import numpy as np
from typing import Optional, Union, Callable
from .base import BaseSpinSystem


class EdwardsAndersonSystem(BaseSpinSystem):
    """
    Edwards-Anderson Model for Spin Glass Systems.
    
    This model implements discrete Ising spins ({-1, 1}) on a d-dimensional lattice
    with quenched random couplings (J_ij). The energy computation handles an arbitrary 
    interaction matrix, which represents the disorder and frustration inherent in 
    spin glass systems.
    
    Args:
        lattice_length (int): The size of each dimension in the lattice.
        lattice_replicas (int): The number of independent replicas to simulate in parallel.
        interaction_matrix (Union[tf.Tensor, np.ndarray]): The specific coupling matrix (J_ij)
            dictating the interaction strength between spins.
        initial_magnetization (float, optional): Sets the probability of spins initializing to +1.
            Defaults to 0.5 (random initialization).
        lattice_dim (int, optional): Number of spatial dimensions. Defaults to 2.
        initial_spin_state (Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]], optional):
            Pre-defined spin states to initialize with. Defaults to None.
    """
    def __init__(
        self,
        lattice_length: int,
        lattice_replicas: int,
        interaction_matrix: Union[tf.Tensor, np.ndarray],
        initial_magnetization: float = 0.5,
        lattice_dim: int = 2,
        initial_spin_state: Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]] = None,
    ):
        self.initial_magnetization = initial_magnetization

        super().__init__(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            initial_spin_state=initial_spin_state
        )

        self.interaction_matrix = self._validate_tensor_shape(
            interaction_matrix,
            expected_shape=tuple(self.shape + self.shape),
            name="Interaction matrix",
        )

    def initialize_state(self) -> tf.Tensor:
        """
        Discrete spins {-1, 1} based on initial magnetization.
        """
        p_up = 0.5 + 0.5 * tf.tanh(self.initial_magnetization)

        # Generate random values for all replicas
        full_shape = [self.lattice_replicas] + self.shape
        rand_vals = tf.random.uniform(full_shape, dtype=tf.float32)

        spin_state = tf.cast(rand_vals < p_up, tf.float32)
        spin_state = 2.0 * spin_state - 1.0

        return spin_state

    # @tf.function
    def compute_energy(self, spin_state: Optional[tf.Variable | tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.spin_state

        # Flatten spins: (replicas, N)
        spin_state_flat = tf.reshape(spin_state, (self.lattice_replicas, -1))

        # Flatten interaction matrix: (N, N)
        interaction_matrix_flat = tf.reshape(
            self.interaction_matrix, (self.number_spins, self.number_spins)
        )

        # E = -0.5 * S^T J S
        # Compute h_local = S @ J  --> shape (replicas, N)
        h_local = tf.matmul(spin_state_flat, interaction_matrix_flat)

        pairwise = -0.5 * tf.reduce_sum(spin_state_flat * h_local, axis=1)

        return pairwise
