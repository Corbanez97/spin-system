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

        updated_energy = self.system.compute_energy(updated)

        return updated, updated_energy

    def _disturb_state(self, num_disturbances: tf.Tensor, theta_max: Optional[tf.Tensor]) -> Tuple[tf.Variable | tf.Tensor, tf.Tensor]:
        if theta_max is None:
            updated, updated_energy = self.flip_spins(num_disturbances)
        else:
            if isinstance(self.system, IsingSystem):
                raise TypeError(
                    "Can't perform rotations on Ising Spins. Remove theta_max or use Spherical System")
            return self.system.spin_state, self.current_energy
        return updated, updated_energy

    # @tf.function
    def step(
        self,
        beta: float,
        num_disturbances: tf.Tensor,
        theta_max: Optional[tf.Tensor] = None
    ) -> 'BaseSpinSystem':
        updated, updated_energy = self._disturb_state(
            num_disturbances=num_disturbances, theta_max=theta_max)

        energy_delta = tf.math.subtract(updated_energy, self.current_energy)

        prob_accept = tf.exp(-tf.multiply(beta, energy_delta))

        random_vals = tf.random.uniform(
            shape=(self.system.lattice_replicas,), dtype=tf.float32)

        accept = tf.logical_or(
            tf.less(energy_delta, 0.0),
            random_vals < prob_accept
        )

        new_spin_state = tf.where(
            tf.reshape(accept, (-1,) + (1,) * self.system.lattice_dim),
            updated,
            self.system.spin_state
        )
        self.system.update_state(new_spin_state)

        self.current_energy = tf.where(
            accept,
            updated_energy,
            self.current_energy
        )
        tf.print(self.current_energy)
        return self.system

    @tf.function
    def sweep(
        self,
        tracker: 'Tracker',
        beta: float,
        num_disturbance: int = 1,
        theta_max: Optional[tf.Tensor] = None,
        sweep_length: Optional[int] = None,
    ) -> 'BaseSpinSystem':
        """
        The orchestrator of multiple steps of the simulation.
        """
        return self.system
