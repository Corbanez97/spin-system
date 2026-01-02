import tensorflow as tf
from typing import Optional, Tuple, Union
from ..models.base import BaseSpinSystem
from ..models.ising import IsingSystem
from ..models.spherical import SphericalSystem
from ..models.z2_gauge import Z2GaugeSystem
from ..measurements.scalars import Energy
from ..measurements.tracker import Tracker
from .base import BaseDynamics


class MetropolisDynamics(BaseDynamics):
    """
    Metropolis-Hastings dynamics driver.
    """

    def step(
        self,
        system: BaseSpinSystem,
        beta: float,
        num_disturb: int = 1,
        theta_max: float = 0.0,
        **kwargs
    ) -> tf.Tensor:
        """
        Perform a single Metropolis update step.
        """
        # Determine method based on system type
        if isinstance(system, IsingSystem):
            return self._flip_spins(system, beta, num_disturb)
        elif isinstance(system, SphericalSystem):
            return self._rotate_spins(system, beta, num_disturb, theta_max)
        elif isinstance(system, Z2GaugeSystem):
            # Placeholder or future implementation
            raise NotImplementedError("Z2 Gauge dynamics not yet implemented.")
        else:
            raise TypeError(f"Unknown system type: {type(system)}")

    def sweep(
        self,
        system: BaseSpinSystem,
        n_steps: int = 100,
        tracker: Optional[Tracker] = None,
        beta: float = 1.0,
        num_disturb: int = 1,
        theta_max: float = 0.0,
        **kwargs
    ):
        """
        Perform a Metropolis sweep.
        """
        # Initialize tracker history if provided
        if tracker is not None:
            history = tracker.init()
            # Initial record
            history = tracker.record(tf.constant(0), system, history)
        else:
            history = {}  # Dummy

        # Loop variables
        i = tf.constant(0)

        # We need to pass 'history' through the loop if tracking is enabled.
        # But tf.while_loop requires consistent structure.
        # If tracker is None, 'history' is empty dict.

        # Define loop body
        def body(i, history_var):
            # Perform update (in-place modification of system.spin_state via assign)
            _ = self.step(system, beta, num_disturb, theta_max)

            # Record
            if tracker is not None:
                new_history = tracker.record(i + 1, system, history_var)
            else:
                new_history = history_var

            return i + 1, new_history

        # Use tf.while_loop
        # Note: system.spin_state is a tf.Variable, so it's updated by side-effect in 'step'.
        # We don't need to pass it as loop var.

        _, final_history = tf.while_loop(
            lambda i, _: i < n_steps,
            body,
            loop_vars=[i, history],
            parallel_iterations=1
        )

        # Finalize
        if tracker is not None:
            return tracker.finalize(final_history)
        else:
            return {}

    @tf.function
    def _flip_spins(
        self,
        system: IsingSystem,
        beta: float,
        num_flips: int
    ) -> tf.Tensor:
        """
        Metropolis update for Ising Model (Spin Flip).
        """
        # 1. Select random spins to flip
        replicas = system.lattice_replicas
        num_spins = tf.cast(system.number_spins, tf.int32)

        # Helper to generate indices (same as legacy)
        idx = tf.stack([
            tf.random.shuffle(tf.range(num_spins))[:num_flips]
            for _ in range(replicas)
        ], axis=0)  # (replicas, num_flips)

        replica_idx = tf.repeat(tf.range(replicas)[:, None], num_flips, axis=1)
        gather_indices = tf.stack([replica_idx, idx], axis=-1)

        spin_flat = tf.reshape(system.spin_state, (replicas, -1))

        # 2. Propose change
        # Flip: s -> -s
        # updates = -s_old
        old_values = tf.gather_nd(spin_flat, gather_indices)
        updates = -old_values

        # Calculate Delta E efficiently
        energy_delta = self._compute_energy_delta_flip(
            system, spin_flat, updates, idx, gather_indices
        )
        # Defensive reshape
        energy_delta = tf.reshape(energy_delta, (replicas,))

        # 3. Acceptance Criterion
        prob_accept = tf.exp(-beta * energy_delta)
        random_vals = tf.random.uniform((replicas,), dtype=tf.float32)
        accept = (energy_delta < 0) | (random_vals < prob_accept)

        # 4. Update State
        # Only update accepted replicas
        # mask indices where accept is True

        # Strategy:
        # Create a full update tensor for indices.
        # But we only apply if accepted.
        # This is tricky with gather_nd logic alone.

        # Easier:
        # Apply updates to a COPY of spin_flat, then use tf.where on the whole replica vector?
        # No, that's expensive for large N.

        # We scatter internal updates efficiently.
        # But we must only scatter for accepted replicas.

        # Filter indices by acceptance
        # accept shape: (replicas,)
        # gather_indices shape: (replicas, num_flips, 2)

        # (replicas, num_flips)
        accepted_mask = tf.gather(accept, gather_indices[..., 0])
        accepted_mask = tf.reshape(accepted_mask, (replicas, num_flips))

        # Flatten everything to scatter
        flat_indices = tf.boolean_mask(gather_indices, accepted_mask)
        flat_updates = tf.boolean_mask(updates, accepted_mask)

        if tf.size(flat_indices) > 0:
            # Apply update
            new_spin_flat = tf.tensor_scatter_nd_update(
                spin_flat, flat_indices, flat_updates
            )
            new_spin_state = tf.reshape(new_spin_flat, system.spin_state.shape)
            system.spin_state.assign(new_spin_state)

        return tf.cast(accept, tf.float32)

    @tf.function
    def _rotate_spins(
        self,
        system: SphericalSystem,
        beta: float,
        num_pairs: int,
        theta_max: float
    ) -> tf.Tensor:
        """
        Metropolis update for Spherical Model (Rotation).
        """
        replicas = system.lattice_replicas
        num_spins = tf.cast(system.number_spins, tf.int32)

        # Select pairs (i, j)
        all_indices = tf.stack([
            tf.random.shuffle(tf.range(num_spins))[:2 * num_pairs]
            for _ in range(replicas)
        ], axis=0)

        idx1 = all_indices[:, :num_pairs]
        idx2 = all_indices[:, num_pairs:]

        replica_idx = tf.repeat(tf.range(replicas)[:, None], num_pairs, axis=1)

        gather_indices_i = tf.stack([replica_idx, idx1], axis=-1)
        gather_indices_j = tf.stack([replica_idx, idx2], axis=-1)

        spin_flat = tf.reshape(system.spin_state, (replicas, -1))

        sigma_i = tf.gather_nd(spin_flat, gather_indices_i)
        sigma_j = tf.gather_nd(spin_flat, gather_indices_j)

        # Propose rotation
        theta = tf.random.uniform(
            [replicas, num_pairs], -theta_max, theta_max
        )
        cos_t, sin_t = tf.cos(theta), tf.sin(theta)

        new_i = cos_t * sigma_i - sin_t * sigma_j
        new_j = sin_t * sigma_i + cos_t * sigma_j

        # Compute deltas
        # Ideally we need the CHANGE in spin values to use the delta formula
        delta_i = new_i - sigma_i
        delta_j = new_j - sigma_j

        # Prepare for delta calculation
        updates = tf.concat([delta_i, delta_j], axis=1)
        disturbed_idx = tf.concat([idx1, idx2], axis=1)

        # Calculate Delta E
        # Note: logic same as flip, using (S_new - S_old)
        energy_delta = self._compute_energy_delta_general(
            system, spin_flat, updates, disturbed_idx
        )
        energy_delta = tf.reshape(energy_delta, (replicas,))

        # Acceptance
        prob_accept = tf.exp(-beta * energy_delta)
        random_vals = tf.random.uniform((replicas,), dtype=tf.float32)
        accept = (energy_delta < 0) | (random_vals < prob_accept)

        # Apply updates
        # Construct absolute new values to scatter
        new_vals = tf.concat([new_i, new_j], axis=1)

        # Expand accept mask to (replicas, 2*num_pairs)
        accept_expanded = tf.repeat(accept[:, None], 2 * num_pairs, axis=1)
        accept_expanded = tf.reshape(
            accept_expanded, (replicas, 2 * num_pairs))

        # Indices for scatter
        scatter_replica_idx = tf.repeat(
            tf.range(replicas)[:, None], 2 * num_pairs, axis=1)
        scatter_indices = tf.stack(
            [scatter_replica_idx, disturbed_idx], axis=-1)

        flat_indices = tf.boolean_mask(scatter_indices, accept_expanded)
        flat_new_vals = tf.boolean_mask(new_vals, accept_expanded)

        if tf.size(flat_indices) > 0:
            new_spin_flat = tf.tensor_scatter_nd_update(
                spin_flat, flat_indices, flat_new_vals
            )
            # Re-normalize if spherical constraint is active?
            # Rotation preserves norm of the PAIR (i, j).
            # If (s_i^2 + s_j^2) is constant, sum s^2 is constant.
            # So explicit re-normalization shouldn't be strictly necessary if numerical error is low.
            # But the constraint check might fail eventually.
            # For now, trust rotation preserves l2 norm.

            new_spin_state = tf.reshape(new_spin_flat, system.spin_state.shape)
            system.spin_state.assign(new_spin_state)

        return tf.cast(accept, tf.float32)

    def _compute_energy_delta_flip(
        self,
        system: BaseSpinSystem,
        spin_flat: tf.Tensor,
        # (New - Old) or just New value? Logic above was updates = -old (which is the NEW value to set? NO)
        replacement_deltas: tf.Tensor,
        # Wait. In flip: s' = -s.  Delta_s = s' - s = -s - s = -2s.
        # My _flip_spins calculated `updates = -old_values`. This is the NEW VALUE to scatter.
        # My _compute_energy_delta helper needs the DELTA (change) or the NEW VALUE?
        # Legacy code `_compute_pairwise_energy_deltas` takes `updated_spin_flat`.
        # I should adapt to be self-contained.

        idx: tf.Tensor,
        gather_indices: tf.Tensor
    ) -> tf.Tensor:
        """
        Computes energy delta for generic sparse updates using interaction matrix.
        Delta E = - sum_{i in changed} (delta_s_i * h_i_old) - 0.5 * sum_{i,j in changed} delta_s_i * J_ij * delta_s_j

        Wait, standard formula:
        E = -0.5 S J S
        E' = -0.5 (S+dS) J (S+dS) = E - 0.5(dS J S + S J dS + dS J dS)
           = E - (dS J S) - 0.5 dS J dS   (using symmetry)
           = E - sum(dS_i * h_i) - 0.5 dS J dS

        So we need dS (change in spin).
        """
        # Calculate dS
        old_val = tf.gather_nd(spin_flat, gather_indices)
        # updates passed in are the NEW VALUES (-old_val)
        new_val = replacement_deltas
        ds = new_val - old_val

        replicas = tf.shape(spin_flat)[0]
        num_spins = tf.shape(system.interaction_matrix)[
            0]  # Flatten shape check needed?

        # Interaction matrix is (N, N) or (dim, ...)?
        # BaseSpinSystem does NOT guarantee flat interaction matrix.
        # subclasses handle it. Ising/Spherical have it.
        # We need to flatten it.
        J_flat = tf.reshape(system.interaction_matrix, (int(
            system.number_spins), int(system.number_spins)))

        # 1. Field term
        # E_field = - sum (S_i * H_i)
        # Delta E_field = - sum (dS_i * H_i)
        if system.external_field is not None:
            H_flat = tf.reshape(system.external_field, (1, -1))
            # (1, num_flips) -> broadcast to (replicas, num_flips)
            H_local = tf.gather(H_flat, idx, axis=1)
            delta_E_field = -tf.reduce_sum(ds * H_local, axis=1)
        else:
            delta_E_field = 0.0

        # 2. Interaction term
        # h_old = S @ J
        h_old = tf.matmul(spin_flat, J_flat)  # (replicas, N)

        # Limit h to disturbed indices
        h_subset = tf.gather(h_old, idx, batch_dims=1)  # (replicas, num_flips)

        term1 = -tf.reduce_sum(ds * h_subset, axis=1)

        # term2 = -0.5 * dS * J_sub * dS
        # J_sub: interactions between flipped spins
        # J_rows = gather rows `idx`. J_sub = gather cols `idx` from J_rows.

        # tf.gather on J_flat (N, N)
        # We need (replicas, num_flips, num_flips) because idx varies by replica.
        # idx is (replicas, num_flips).

        # J is constant across replicas.
        # But indices are different.
        # Use gather with batch_dims=0 ? No, indices are batched.

        # J_flat is (N, N).
        # We want J[idx[r, :], idx[r, :]] for each r.

        # Expand J to (replicas, N, N) ? Expensive.
        # Use gather on axis 0 first?
        # J_rows = tf.gather(J_flat, idx) -> (replicas, num_flips, N) ??
        # tf.gather(params, indices). params=(N,N). indices=(R, k).
        # Result: (R, k, N). Yes!

        J_rows = tf.gather(J_flat, idx)

        # Now gather columns.
        # We want cols `idx` from J_rows.
        # J_rows is (R, k, N). indices is (R, k).
        # We want result (R, k, k).
        # effectively: result[r, i, j] = J_rows[r, i, idx[r, j]]

        J_sub = tf.gather(J_rows, idx, axis=2, batch_dims=1)

        # Quad form: dS (R, k). J_sub (R, k, k).
        # Expand dS -> (R, 1, k) and (R, k, 1)

        ds_expanded = tf.expand_dims(ds, axis=1)  # (R, 1, k)

        quad = tf.matmul(ds_expanded, tf.matmul(
            J_sub, tf.expand_dims(ds, axis=-1)))
        # (R, 1, 1)

        term2 = -0.5 * tf.squeeze(quad, axis=[1, 2])

        return delta_E_field + term1 + term2

    def _compute_energy_delta_general(
        self,
        system: BaseSpinSystem,
        spin_flat: tf.Tensor,
        dS: tf.Tensor,  # The change values
        disturbed_idx: tf.Tensor  # (replicas, num_changed)
    ) -> tf.Tensor:
        """
        Same as flip but accepts pre-calculated dS and indices.
        Used for rotation where dS is not just -2*S.
        """
        # Reuse logic
        # 1. Field
        if system.external_field is not None:
            H_flat = tf.reshape(system.external_field, (-1,))
            H_local = tf.gather(H_flat, disturbed_idx)  # (replicas, num_flips)
            delta_E_field = -tf.reduce_sum(dS * H_local, axis=1)
        else:
            delta_E_field = 0.0

        # 2. Interaction
        J_flat = tf.reshape(system.interaction_matrix, (int(
            system.number_spins), int(system.number_spins)))

        h_old = tf.matmul(spin_flat, J_flat)
        h_subset = tf.gather(h_old, disturbed_idx, batch_dims=1)

        term1 = -tf.reduce_sum(dS * h_subset, axis=1)

        J_rows = tf.gather(J_flat, disturbed_idx)
        J_sub = tf.gather(J_rows, disturbed_idx, axis=2, batch_dims=1)

        ds_expanded = tf.expand_dims(dS, axis=1)
        quad = tf.matmul(ds_expanded, tf.matmul(
            J_sub, tf.expand_dims(dS, axis=-1)))

        term2 = -0.5 * tf.squeeze(quad, axis=[1, 2])

        return delta_E_field + term1 + term2
