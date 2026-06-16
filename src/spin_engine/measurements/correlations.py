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


class OverlapDistribution(OverlapMatrix):
    """
    Computes the distribution of off-diagonal overlaps P(q).
    Uses the OverlapMatrix to extract Q_ab for a != b.

    Returns: 1D Tensor of shape (replicas * (replicas - 1) / 2)
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        q_matrix = super().compute(spin_state, system)
        replicas = q_matrix.shape[0]

        # Create a boolean mask for the strictly upper triangular part
        mask = tf.linalg.band_part(tf.ones((replicas, replicas), dtype=tf.bool), 0, -1)
        mask = tf.linalg.set_diag(mask, tf.zeros(replicas, dtype=tf.bool))

        return tf.boolean_mask(q_matrix, mask)


class ParisiOverlapParameter(OverlapDistribution):
    """
    Computes the Parisi overlap parameter: <q^2> - <|q|>^2
    This serves as an indicator of replica symmetry breaking (RSB).

    Returns: Scalar Tensor
    """

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None,
                system: Optional['BaseSpinSystem'] = None) -> tf.Tensor:

        q_values = super().compute(spin_state, system)

        q_sq_mean = tf.reduce_mean(tf.square(q_values))
        abs_q_mean = tf.reduce_mean(tf.abs(q_values))

        return q_sq_mean - tf.square(abs_q_mean)
