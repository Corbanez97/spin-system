import tensorflow as tf
from .base import Dynamics

from typing import Optional, TYPE_CHECKING, Tuple, cast

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
        self.current_energy = tf.Variable(
            system.compute_energy(), trainable=False, name="current_energy")

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

    def _disturb_state(self, num_disturbances: tf.Tensor, theta_max: Optional[tf.Tensor]) -> Tuple[tf.Variable | tf.Tensor, tf.Variable | tf.Tensor]:
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
    ) -> None:
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

        rank = tf.rank(self.system.spin_state)
        target_shape = tf.concat(
            [[-1], tf.ones([tf.add(rank, -1)], dtype=tf.int32)], axis=0)

        new_spin_state = tf.where(
            tf.reshape(accept, target_shape),
            updated,
            self.system.spin_state
        )
        self.system.update_state(new_spin_state)

        new_energy = tf.where(
            accept,
            updated_energy,
            self.current_energy
        )
        self.current_energy.assign(new_energy)
        return None

    # TODO: Fix typing errors here...

    @tf.function
    def sweep(
        self,
        tracker: 'Tracker',
        beta: float,
        sweep_length: int,
        num_disturbances:  tf.Tensor = cast(tf.Tensor, 1),
        theta_max: Optional[tf.Tensor] = None,
    ) -> None:
        """
        The orchestrator of multiple steps of the simulation.
        """
        tracking_arrays = tracker.init_run(cast(tf.Tensor, sweep_length))

        tracking_arrays = tracker.track(
            cast(tf.Tensor, 0), self.system, tracking_arrays)

        def body(i, tracking_arrays):
            _ = self.step(beta, cast(tf.Tensor, num_disturbances), theta_max)

            current_step = i + 1
            new_arrays = tracker.track(
                current_step, self.system, tracking_arrays)

            return i + 1, new_arrays

        i0 = tf.constant(0, dtype=tf.int32)
        loop_result = tf.while_loop(
            cond=lambda i, _: i < sweep_length,
            body=body,
            loop_vars=[i0, tracking_arrays]
        )

        final_arrays = cast(Tuple, loop_result)[1]
        tracker.finalize(final_arrays)

        return None
