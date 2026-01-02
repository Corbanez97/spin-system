import tensorflow as tf
from typing import Optional, Any
from .base import Measurement


class Energy(Measurement):
    """
    Computes the total energy of the system.
    Returns a tensor of shape (replicas,).
    """
    def compute(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        return self.system.compute_energy(spin_state)


class Magnetization(Measurement):
    """
    Computes the average magnetization per site for each replica.
    Returns a tensor of shape (replicas,).
    """
    def compute(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.system.spin_state.value()

        # Match legacy: compute_magnetizations
        # tf.reduce_mean(tf.reshape(spin_state, (self.lattice_replicas, -1)), axis=1)

        flat_state = tf.reshape(spin_state, (self.system.lattice_replicas, -1))
        return tf.reduce_mean(flat_state, axis=1)


class MagneticSusceptibility(Measurement):
    """
    Computes the variance of the spin state (Legacy implementation).
    Returns a scalar.
    """
    def compute(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        if spin_state is None:
            spin_state = self.system.spin_state.value()

        # Match legacy: compute_magnetic_susceptibility
        # tf.math.reduce_variance(spin_state)
        return tf.math.reduce_variance(spin_state)
