import tensorflow as tf
from typing import Optional
from .base import Measurement


class OverlapMatrix(Measurement):
    """
    Computes the overlap matrix between all replicas.
    Q_ab = (1/N) * sum_i s_i^a * s_i^b
    Returns: Tensor of shape (replicas, replicas)
    """

    def compute(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.system.spin_state.value()

        spin_flat = tf.reshape(spin_state, (self.system.lattice_replicas, -1))

        # We need number_spins. In BaseSpinSystem (or subclasses) it should be available.
        # Check if BaseSpinSystem has number_spins.
        # legacy_core.py line 38: self.number_spins = tf.cast(lattice_length ** lattice_dim, tf.float32)
        # Assuming BaseSpinSystem has it or can compute it.
        # Let's rely on self.system.number_spins if available, or compute from shape.

        # Safe access to number_spins, assuming it's a property or attribute
        if hasattr(self.system, 'number_spins'):
            N = self.system.number_spins
        else:
            N = tf.cast(tf.reduce_prod(self.system.shape), tf.float32)

        overlap = tf.matmul(spin_flat, spin_flat, transpose_b=True)
        overlap /= N

        return overlap
