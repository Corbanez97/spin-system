import tensorflow as tf
from typing import Optional, TYPE_CHECKING
from .base import Measurement

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem


class Energy(Measurement):
    """
    Computes the total energy of the system.
    Requires a system instance to access the Hamiltonian logic.
    Returns a tensor of shape (replicas,).
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        state, sys = self._resolve(spin_state, system)

        if sys is None:
            raise ValueError(
                "Energy computation requires a system (Hamiltonian logic).")

        return sys.compute_energy(state)


class Magnetization(Measurement):
    """
    Computes the average magnetization per site for each replica.
    Can function with just a spin_state (inferring replicas from shape).
    Returns a tensor of shape (replicas,).
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        state, sys = self._resolve(spin_state, system)

        # Resolve replicas: System metadata > Tensor shape
        replicas = sys.lattice_replicas if sys else state.shape[0]

        flat_state = tf.reshape(state, (replicas, -1))
        return tf.reduce_mean(flat_state, axis=1)


class MagneticSusceptibility(Measurement):
    """
    Computes the variance of the spin state magnetizations across replicas.
    Can function with just a spin_state.
    Returns a scalar.
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        state, sys = self._resolve(spin_state, system)

        replicas = sys.lattice_replicas if sys else state.shape[0]

        flat_state = tf.reshape(state, (replicas, -1))
        m_per_replica = tf.reduce_mean(flat_state, axis=1)

        return tf.math.reduce_variance(m_per_replica)
