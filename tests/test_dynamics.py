import pytest
import tensorflow as tf
from typing import cast

from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.models import IsingSystem
from spin_engine.dynamics import MetropolisHastings


class TestDynamics:

    @pytest.fixture
    def setup_system(self):
        lattice_dim = 2
        lattice_length = 4
        lattice_replicas = 16

        interaction_matrix = PeriodicNearestNeighborInteraction().generate(
            lattice_dim, lattice_length)

        ising_system = IsingSystem(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix
        )

        simulation = MetropolisHastings(ising_system)

        return {
            'system': ising_system,
            'simulation': simulation,
            'replicas': lattice_replicas,
            'num_spins': lattice_length ** lattice_dim
        }

    def test_flip_spins_changes_sign(self, setup_system):
        simulation = setup_system['simulation']
        system = setup_system['system']
        replicas = setup_system['replicas']

        initial_spins = tf.identity(system.spin_state)

        num_flips = cast(tf.Tensor, tf.constant(1))

        # We need to capture which spins were flipped to verify exact sign change
        # The current API returns proposed_spin_state, energy_delta
        proposed_spin_state, _ = simulation.flip_spins(num_flips=num_flips)

        # Check that exactly num_flips * replicas spins changed
        # Since we use -1/1 spins, changed spins will have product -1, unchanged 1
        product = initial_spins * proposed_spin_state

        # Count differences
        diff = tf.reduce_sum(
            tf.cast(tf.abs(initial_spins - proposed_spin_state) > 1e-5, tf.int32))

        expected_diff = replicas * num_flips
        assert diff == expected_diff, f"Expected {expected_diff} flips, got {diff}"

        # Verify the changed ones are exactly negated
        # Where they differ, sum should be 0 (x + (-x) = 0)
        # Or diff should be 2.0 or -2.0

        changes = proposed_spin_state - initial_spins
        # Filter non-zero changes
        nonzero_changes = tf.boolean_mask(changes, tf.abs(changes) > 0)

        # Changes should be magnitude 2 (flip from 1 to -1 or -1 to 1)
        tf.debugging.assert_near(tf.abs(nonzero_changes), 2.0)

    def test_energy_delta_small_system(self):
        # Setup small system for manual verification
        lattice_dim = 2
        lattice_length = 4  # Small enough
        lattice_replicas = 1

        interaction_matrix = PeriodicNearestNeighborInteraction().generate(
            lattice_dim, lattice_length)

        ising_system = IsingSystem(
            lattice_dim=lattice_dim,
            lattice_length=lattice_length,
            lattice_replicas=lattice_replicas,
            interaction_matrix=interaction_matrix
        )

        simulation = MetropolisHastings(ising_system)

        initial_energy = simulation.current_energy
        num_flips = cast(tf.Tensor, tf.constant(1))

        proposed_spin_state, energy_delta = simulation.flip_spins(
            num_flips=num_flips)

        # Calculate new energy manually from the proposed state
        # We can't trust the incremental update here, we must recompute from scratch
        # IsingSystem doesn't have compute_energy generic method attached to it for arbitrary state
        # based on previous file reads, it seemed to, checking MetropolisHastings:
        # self.current_energy = system.compute_energy()
        # and: energy_delta = ... self.system.compute_energy(updated)
        # So IsingSystem must have compute_energy(spin_state=...)

        new_energy_recomputed = ising_system.compute_energy(
            spin_state=proposed_spin_state)

        actual_delta = new_energy_recomputed - initial_energy

        tf.debugging.assert_near(energy_delta, actual_delta, atol=1e-5)
