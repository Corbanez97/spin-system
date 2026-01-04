import tensorflow as tf
from typing import Optional, Any
from .base import Measurement


class Energy(Measurement):
    """
    Computes the total energy of the system.
    Returns a tensor of shape (replicas,).
    """
    # @tf.function

    def compute(self, spin_state: Optional[tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:
        if system is None:
            system = self.system
        return system.compute_energy(spin_state)


class Magnetization(Measurement):
    """
    Computes the average magnetization per site for each replica.
    Returns a tensor of shape (replicas,).
    """

    def compute(self, spin_state: Optional[tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:
        if system is None:
            system = self.system

        if spin_state is None:
            spin_state = system.spin_state.value()

        # Match legacy: compute_magnetizations
        # tf.reduce_mean(tf.reshape(spin_state, (self.lattice_replicas, -1)), axis=1)

        flat_state = tf.reshape(spin_state, (system.lattice_replicas, -1))
        return tf.reduce_mean(flat_state, axis=1)


class MagneticSusceptibility(Measurement):
    """
    Computes the variance of the spin state (Legacy implementation).
    Returns a scalar.
    """

    def compute(self, spin_state: Optional[tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:
        if system is None:
            system = self.system

        if spin_state is None:
            spin_state = system.spin_state.value()

        # Compute magnetization for each replica
        if system is not None:
            replicas = system.lattice_replicas
        else:
            replicas = tf.shape(spin_state)[0]

        flat_state = tf.reshape(spin_state, (replicas, -1))
        # Magnetization per replica
        magnetizations = tf.reduce_mean(flat_state, axis=1)

        # Susceptibility: Variance of the magnetizations across replicas
        return tf.math.reduce_variance(magnetizations)
