import tensorflow as tf
from typing import Optional, TYPE_CHECKING
from .base import Measurement

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem


class OverlapMatrix(Measurement):
    """
    Computes the overlap matrix between all replicas.
    Q_ab = (1/N) * sum_i s_i^a * s_i^b

    Returns: Tensor of shape (replicas, replicas)
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        state, sys = self._resolve(spin_state, system)

        replicas = sys.lattice_replicas if sys else state.shape[0]

        if sys is not None:
            if hasattr(sys, 'number_spins'):
                n_spins = tf.cast(sys.number_spins, tf.float32)
            else:
                n_spins = tf.cast(tf.reduce_prod(sys.shape), tf.float32)
        else:
            n_spins = tf.cast(tf.reduce_prod(state.shape[1:]), tf.float32)

        spin_flat = tf.reshape(state, (replicas, -1))

        overlap = tf.matmul(spin_flat, spin_flat, transpose_b=True)

        return overlap / n_spins
