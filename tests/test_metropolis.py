import tensorflow as tf
import pytest
import numpy as np
from spin_engine.models.ising import IsingSystem
from spin_engine.models.spherical import SphericalSystem
from spin_engine.dynamics.metropolis import MetropolisDynamics
from spin_engine.measurements.scalars import Energy, Magnetization
from spin_engine.measurements.tracker import Tracker


class TestMetropolisDynamics:

    @pytest.fixture
    def dynamics(self):
        return MetropolisDynamics()

    def test_ising_minimization(self, dynamics):
        # 2x2 Ferromagnetic Ising
        J = tf.ones((4, 4), dtype=tf.float32)  # All-to-all ferromagnetic
        # Actually simplified: 1D chain with PBC
        # Let's use internal validation logic or just run for long time at Beta=100 (low temp)

        system = IsingSystem(lattice_length=5, lattice_replicas=1,
                             # Dummy J
                             lattice_dim=1, interaction_matrix=tf.eye(5))
        # Better J: simple neighbor

        # Manually construct J for 1D chain length 4 PBC
        # 0-1, 1-2, 2-3, 3-0
        # shape 4x4
        indices = [[0, 1], [1, 2], [2, 3], [
            3, 0], [1, 0], [2, 1], [3, 2], [0, 3]]
        values = [1.0] * 8
        J = tf.scatter_nd(indices, values, (4, 4))

        system = IsingSystem(4, 1, J, lattice_dim=1)

        # High beta = Low Temp -> Should align
        beta = 10.0

        # Run sweep
        dynamics.sweep(system, n_steps=100, beta=beta, num_disturb=1)

        # Check energy is low.
        # Max alignment: all +1 or all -1.
        # E = -0.5 * sum(s_i s_j J_ij)
        # 4 bonds * 1 * 1 = 4.  -0.5 * 4 * 2 (symmetric) = -4.0

        final_energy = system.compute_energy()
        assert np.isclose(final_energy.numpy(), -4.0)

    def test_spherical_minimization(self, dynamics):
        # Spherical model
        system = SphericalSystem(
            lattice_length=4,
            lattice_replicas=1,
            interaction_matrix=tf.eye(4),  # Self-interaction only? Trivial.
            lattice_dim=1,
            spherical_constraint=True
        )
        # Use simpler J: diagonal 1. Ideal state: align to self?
        # Energy = -0.5 * sum S_i * S_i = -0.5 * sum(S^2)
        # Constraint sum(S^2) = N = 4.
        # E = -0.5 * 4 = -2.0. Constant energy surface for J=I?
        # Yes.

        # Let's use non-trivial J.
        indices = [[0, 1], [1, 0]]
        values = [1.0, 1.0]
        J = tf.scatter_nd(indices, values, (4, 4))

        system = SphericalSystem(
            4, 1, J, lattice_dim=1, spherical_constraint=True)

        # Run
        dynamics.sweep(system, n_steps=200, beta=100.0,
                       num_disturb=1, theta_max=0.5)

        # Check that constraint is maintained (roughly)
        # Note: Metropolis rotate preserves pair norm.
        # If started valid, should stay valid.
        s = system.spin_state
        norm = tf.reduce_sum(s**2)
        assert np.allclose(norm.numpy(), 4.0, atol=1e-4)

    def test_dispatch_logic(self, dynamics):
        # Correctly initialized 1D systems
        ising = IsingSystem(4, 1, tf.zeros((4, 4)), lattice_dim=1)
        spherical = SphericalSystem(4, 1, tf.zeros((4, 4)), lattice_dim=1)

        # Calling step(ising) should call flip
        # Calling step(spherical) should call rotate

        # Verify step works
        dynamics.step(ising, beta=1.0)
        dynamics.step(spherical, beta=1.0, theta_max=0.1)

        # Test error if we pass junk
        with pytest.raises(TypeError):
            dynamics.step("NotASystem", beta=1.0)

    def test_tracker_integration(self, dynamics):
        sys = IsingSystem(4, 1, tf.zeros((4, 4)), lattice_dim=1)
        tracker = Tracker([Energy(sys)], sweep_length=10, granularity=1)

        results = dynamics.sweep(sys, n_steps=10, tracker=tracker)
        assert "Energy" in results
        assert results["Energy"].shape[0] == 11
