import tensorflow as tf
import numpy as np
from typing import Optional, Union, Callable, cast
from .base import BaseSpinSystem


class SphericalSystem(BaseSpinSystem):
    def __init__(
        self,
        lattice_length: int,
        lattice_replicas: int,
        interaction_matrix: Union[tf.Tensor, np.ndarray],
        external_field: Optional[Union[tf.Tensor, np.ndarray]] = None,
        initial_magnetization: float = 0.0,
        spherical_constraint: bool = False,
        lattice_dim: int = 2,
        initial_spin_state: Optional[Union[tf.Tensor,
                                           Callable[[], tf.Tensor]]] = None,
    ):
        self.initial_magnetization = initial_magnetization
        self.spherical_constraint = spherical_constraint

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
        Continuous spins normally distributed.
        """
        full_shape = [self.lattice_replicas] + self.shape
        spin_state = tf.random.normal(
            full_shape, mean=self.initial_magnetization, stddev=1.0
        )

        if self.spherical_constraint:
            spin_state = cast(tf.Tensor, self._apply_spherical_constraint(
                spin_state))  # Casting because I know this is a Tensor

        return spin_state

    @tf.function
    def _apply_spherical_constraint(self, spin_state: tf.Tensor) -> tf.Tensor:
        """
        Applies the spherical constraint independently to each replica.
        Sum(s_i^2) = N
        """
        original_shape = tf.shape(spin_state)
        spin_state_flat_replicas = tf.reshape(
            spin_state, (self.lattice_replicas, -1)
        )

        normalized_flat = tf.math.l2_normalize(
            spin_state_flat_replicas, axis=1
        )

        normalized_spins = tf.reshape(normalized_flat, original_shape)

        return tf.sqrt(self.number_spins) * normalized_spins

    # @tf.function
    def compute_energy(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.spin_state.value()
        # Flatten spins: (replicas, N)
        spin_state_flat = tf.reshape(
            self.spin_state, (self.lattice_replicas, -1))

        # Flatten interaction matrix: (N, N)
        interaction_matrix_flat = tf.reshape(
            self.interaction_matrix, (self.number_spins, self.number_spins)
        )

        # Flatten field
        external_field_flat = tf.reshape(self.external_field, (1, -1))

        h_local = tf.matmul(spin_state_flat, interaction_matrix_flat)

        pairwise = -0.5 * tf.reduce_sum(spin_state_flat * h_local, axis=1)
        field_term = -tf.reduce_sum(spin_state_flat *
                                    external_field_flat, axis=1)

        return pairwise + field_term
