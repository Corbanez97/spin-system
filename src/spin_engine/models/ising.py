import tensorflow as tf
import numpy as np
from typing import Optional, Union, Callable
from .base import BaseSpinSystem


class IsingSystem(BaseSpinSystem):
    def __init__(
        self,
        lattice_length: int,
        lattice_replicas: int,
        interaction_matrix: Union[tf.Tensor, np.ndarray],
        external_field: Optional[Union[tf.Tensor, np.ndarray]] = None,
        initial_magnetization: float = 0.5,
        lattice_dim: int = 2,
        initial_spin_state: Optional[Union[tf.Tensor,
                                           Callable[[], tf.Tensor]]] = None,
    ):
        self.initial_magnetization = initial_magnetization

        # Validate/Store interaction matrix and field BEFORE calling super().__init__
        # because initialize_state might rely on them (though here it only uses magnetization)

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

        self.external_field = self._validate_tensor_shape(
            external_field,
            expected_shape=tuple(self.shape),
            name="External field",
            allow_none=True,
            default=tf.zeros(self.shape, dtype=tf.float32),
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
    def compute_energy(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.spin_state.value()

        # Flatten spins: (replicas, N)
        spin_state_flat = tf.reshape(spin_state, (self.lattice_replicas, -1))

        # Flatten interaction matrix: (N, N)
        interaction_matrix_flat = tf.reshape(
            self.interaction_matrix, (self.number_spins, self.number_spins)
        )

        # Flatten field: (1, N) -> elementwise multiply broadcasts over replicas
        external_field_flat = tf.reshape(self.external_field, (1, -1))

        # E = -0.5 * S^T J S - h S
        # Compute h_local = S @ J  --> shape (replicas, N)
        h_local = tf.matmul(spin_state_flat, interaction_matrix_flat)

        pairwise = -0.5 * tf.reduce_sum(spin_state_flat * h_local, axis=1)
        field_term = -tf.reduce_sum(spin_state_flat *
                                    external_field_flat, axis=1)

        return pairwise + field_term
