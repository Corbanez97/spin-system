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
    def compute_energy(self, spin_state: Optional[tf.Variable | tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.spin_state

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

    def compute_delta_energy(
        self,
        spin_state: tf.Tensor,
        updated_spin_state: tf.Tensor,
        changed_indices: tf.Tensor,
    ) -> tf.Tensor:
        """
        Efficient ΔE for Ising spin flips.

        For Ising flips at sites D with Δσ_n = -2σ_n:
            ΔE = 2 Σ_k σ_{n_k} (h_{n_k} + h^ext_{n_k})
                 - 2 Σ_{i,j∈D} J_{ij} σ_i σ_j   (cross-term for multi-flip)

        For single-site flips (num_flips=1), the cross-term vanishes (J_nn=0).

        Complexity: O(replicas × num_flips × N) — reads num_flips rows of J.
        Compare to compute_energy: O(replicas × N²).

        Args:
            spin_state: Current spin state, shape (replicas, L, ..., L).
            updated_spin_state: Proposed spin state (unused, indices suffice).
            changed_indices: Flat indices of flipped sites, shape (replicas, num_flips).

        Returns:
            tf.Tensor of shape (replicas,) — the energy difference E_new - E_old.
        """
        N = tf.cast(self.number_spins, tf.int32)
        num_flips = tf.shape(changed_indices)[1]

        # Flatten spins: (replicas, N)
        spin_flat = tf.reshape(spin_state, (self.lattice_replicas, -1))

        # Flatten J: (N, N)
        J_flat = tf.reshape(self.interaction_matrix, (N, N))

        # Gather J rows for all flipped sites across all replicas
        # Indices may differ per replica, so we flatten and reshape
        flat_idx = tf.reshape(changed_indices, [-1])              # (replicas * num_flips,)
        J_rows = tf.gather(J_flat, flat_idx)                      # (replicas * num_flips, N)
        J_rows = tf.reshape(J_rows, (self.lattice_replicas, num_flips, N))

        # Local fields h_n = Σ_j J_{nj} σ_j for each flipped site
        # spin_flat: (replicas, N) → (replicas, 1, N) for broadcasting
        h_local = tf.reduce_sum(
            J_rows * spin_flat[:, tf.newaxis, :], axis=-1
        )  # (replicas, num_flips)

        # Spin values at flipped sites: (replicas, num_flips)
        s_flipped = tf.gather(spin_flat, changed_indices, batch_dims=1)

        # External field at flipped sites
        field_flat = tf.reshape(self.external_field, [-1])        # (N,)
        h_ext_flipped = tf.gather(field_flat, flat_idx)           # (replicas * num_flips,)
        h_ext_flipped = tf.reshape(h_ext_flipped, (self.lattice_replicas, num_flips))

        # Term 1: Σ_k 2 * σ_{n_k} * (h_{n_k} + h^ext_{n_k})
        term1 = tf.reduce_sum(
            2.0 * s_flipped * (h_local + h_ext_flipped), axis=-1
        )  # (replicas,)

        # Term 2: Cross-term for multi-site flips
        # -½ Σ_{i,j∈D} J_ij Δσ_i Δσ_j where Δσ = -2σ
        # = -½ Σ_{i,j∈D} J_ij (-2σ_i)(-2σ_j) = -2 Σ_{i,j∈D} J_ij σ_i σ_j
        # For single flips J_nn=0, so this vanishes automatically.
        J_sub = tf.gather(J_rows, changed_indices, batch_dims=1, axis=2)  # (replicas, num_flips, num_flips)
        term2 = -2.0 * tf.reduce_sum(
            s_flipped[:, :, tf.newaxis] * J_sub * s_flipped[:, tf.newaxis, :],
            axis=[-2, -1]
        )  # (replicas,)

        return term1 + term2
