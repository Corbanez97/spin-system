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

        new_spin_state = tf.where(
            tf.reshape(accept, (-1,) + (1,) * self.system.lattice_dim),
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

    @tf.function
    def sweep(
        self,
        tracker: 'Tracker',
        beta: float,
        num_disturbances: int = 1,
        theta_max: Optional[tf.Tensor] = None,
        sweep_length: Optional[int] = None,
    ) -> None:
        """
        The orchestrator of multiple steps of the simulation.
        """
        if sweep_length is None:
            spin_float = tf.cast(self.system.number_spins, tf.float32)
            granularity_float = tf.cast(tracker.granularity, tf.float32)
            sweep_length = tf.cast(spin_float * granularity_float, tf.int32)
        else:
            sweep_length = tf.cast(sweep_length, tf.int32)

        # Initialize tracking (returns dict of TensorArrays)
        tracking_arrays = tracker.init_run(sweep_length)

        # Track initial state (step 0)
        # Note: step is 0-indexed.
        tracking_arrays = tracker.track(tf.constant(
            0, dtype=tf.int32), self.system, tracking_arrays)

        def body(i, tracking_arrays):
            # Perform step
            _ = self.step(beta, num_disturbances, theta_max)

            # Track current step (i + 1)
            # if i=0 (first iteration), we just finished step 1.
            current_step = i + 1
            new_arrays = tracker.track(
                current_step, self.system, tracking_arrays)

            return i + 1, new_arrays

        # Loop
        i0 = tf.constant(0, dtype=tf.int32)
        _, final_arrays = tf.while_loop(
            lambda i, _: i < sweep_length,
            body,
            loop_vars=[i0, tracking_arrays]
        )

        # Finalize tracking (stores to tracker.history)
        tracker.finalize(final_arrays)

        return None
