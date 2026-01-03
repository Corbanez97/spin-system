import tensorflow as tf
from .base import Dynamics

from typing import Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem
    from spin_engine.models.ising import IsingSystem
    from spin_engine.dynamics.tracker import Tracker


class MetropolisHastings(Dynamics):
    """
    Metropolis-Hastings dynamics for the Spin System
    """

    def __init__(self, system: 'BaseSpinSystem') -> None:
        super().__init__(system)
        self.current_energy = system.compute_energy()

    def flip_spins(self, num_flips: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Flip n spins 
        """
        spin_flat = tf.reshape(self.system.spin_state,
                               (self.system.lattice_replicas, -1))

        idx = tf.stack([
            tf.random.shuffle(tf.range(self.system.number_spins, dtype=tf.int32))[
                :num_flips]
            for _ in range(self.system.lattice_replicas)
        ], axis=0)
        replica_idx = tf.repeat(tf.range(self.system.lattice_replicas)[
                                :, None], num_flips, axis=1)
        scatter_indices = tf.stack([replica_idx, idx], axis=-1)
        scatter_indices = tf.reshape(scatter_indices, (-1, 2))

        updates = tf.reshape(
            -tf.gather_nd(spin_flat, scatter_indices),
            [-1]
        )

        updated = tf.tensor_scatter_nd_update(
            spin_flat, scatter_indices, updates)
        updated = tf.reshape(updated, self.system.spin_state.shape)

        energy_delta = tf.math.subtract(
            self.system.compute_energy(updated), self.current_energy)

        return updated, energy_delta

    def _disturb_state(self, num_disturb: tf.Tensor, theta_max: Optional[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        if theta_max is None:
            updated, energy_delta = self.flip_spins(num_disturb)
        else:
            if isinstance(self.system, IsingSystem):
                raise TypeError(
                    "Can't perform rotations on Ising Spins. Remove theta_max or use Spherical System")
            return self.system.spin_state.value(), self.current_energy
        return updated, energy_delta

    @tf.function
    def step(
        self,
        beta: float,
        numb_disturbances: tf.Tensor,
        theta_max: Optional[tf.Tensor] = None
    ) -> 'BaseSpinSystem':

        return self.system

    @tf.function
    def sweep(
        self,
        tracker: 'Tracker',
        beta: float,
        num_disturb: int = 1,
        theta_max: Optional[tf.Tensor] = None,
        sweep_length: Optional[int] = None,
    ) -> 'BaseSpinSystem':
        """
        The orchestrator of multiple steps of the simulation.
        """
        return self.system
