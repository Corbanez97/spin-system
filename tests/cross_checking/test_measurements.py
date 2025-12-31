from spin_engine.measurements import (
    Energy, Magnetization, MagneticSusceptibility, OverlapMatrix,
    Plaquette, WilsonLoop
)
from spin_engine.models.z2_gauge import Z2GaugeSystem
from spin_engine.models.spherical import SphericalSystem
from spin_engine.models.ising import IsingSystem
from legacy_core import SpinSystem as LegacySpinSystem
import tensorflow as tf
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../src')))
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


class TestMeasurementsCrossCheck:

    @pytest.fixture
    def setup_data(self):
        L = 4
        replicas = 2
        dim = 2
        N = L**dim

        # Consistent random seed
        tf.random.set_seed(42)

        # J must be (L, L, L, L) for 2D Ising/Spherical
        # We start with (N, N) symmetric and reshape

        # Note: legacy code flattens (L, L, L, L) to (N, N) internally.
        # But for shape validation it expects (L, L, L, L).

        J_flat = tf.random.normal((N, N))
        J_flat = 0.5 * (J_flat + tf.transpose(J_flat))  # Symmetric
        J = tf.reshape(J_flat, (L, L, L, L))

        h = tf.random.normal((L, L))

        # Initial spin state
        initial_spins_ising = tf.where(
            tf.random.uniform((replicas, L, L)) > 0.5,
            1.0, -1.0
        )

        initial_spins_spherical = tf.random.normal((replicas, L, L))

        return {
            'L': L, 'replicas': replicas, 'dim': dim, 'N': N,
            'J': J, 'h': h,
            'spins_ising': initial_spins_ising,
            'spins_spherical': initial_spins_spherical
        }

    def test_ising_measurements(self, setup_data):
        """Cross-check Ising measurements against legacy implementation."""
        L, replicas, J, h = setup_data['L'], setup_data['replicas'], setup_data['J'], setup_data['h']
        spins = setup_data['spins_ising']

        legacy = LegacySpinSystem(
            lattice_dim=2,
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=J,
            initial_spin_state=spins,
            external_field=h,
            model="ising"
        )

        new_system = IsingSystem(
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=J,
            external_field=h,
            initial_spin_state=spins
        )

        # 1. Energy
        legacy_energy = legacy.compute_pairwise_energies()
        new_energy = Energy(new_system).compute()
        tf.debugging.assert_near(legacy_energy, new_energy, atol=1e-5)

        # 2. Magnetization
        legacy_mag = legacy.compute_magnetizations()
        new_mag = Magnetization(new_system).compute()
        tf.debugging.assert_near(legacy_mag, new_mag, atol=1e-5)

        # 3. Magnetic Susceptibility
        legacy_susp = legacy.compute_magnetic_susceptibility()
        new_susp = MagneticSusceptibility(new_system).compute()
        tf.debugging.assert_near(legacy_susp, new_susp, atol=1e-5)

        # 4. Overlap Matrix
        legacy_overlap = legacy.compute_overlap_matrix()
        new_overlap = OverlapMatrix(new_system).compute()
        tf.debugging.assert_near(legacy_overlap, new_overlap, atol=1e-5)

    def test_spherical_measurements(self, setup_data):
        """Cross-check Spherical measurements against legacy implementation."""
        L, replicas, J, h = setup_data['L'], setup_data['replicas'], setup_data['J'], setup_data['h']
        spins = setup_data['spins_spherical']

        legacy = LegacySpinSystem(
            lattice_dim=2,
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=J,
            initial_spin_state=spins,
            external_field=h,
            model="spherical",
            spherical_constraint=False  # Testing unconstrained first to match exact spins
        )

        new_system = SphericalSystem(
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=J,
            external_field=h,
            initial_spin_state=spins,
            spherical_constraint=False
        )

        # 1. Energy
        legacy_energy = legacy.compute_pairwise_energies()
        new_energy = Energy(new_system).compute()
        tf.debugging.assert_near(legacy_energy, new_energy, atol=1e-5)

        # 2. Magnetization
        legacy_mag = legacy.compute_magnetizations()
        new_mag = Magnetization(new_system).compute()
        tf.debugging.assert_near(legacy_mag, new_mag, atol=1e-5)

        # 3. Magnetic Susceptibility
        legacy_susp = legacy.compute_magnetic_susceptibility()
        new_susp = MagneticSusceptibility(new_system).compute()

        # Use simple assert_near or assert_equal
        tf.debugging.assert_near(legacy_susp, new_susp, atol=1e-5)

    def test_gauge_params_errors(self, setup_data):
        """Ensure WilsonLoop raises TypeError for non-Z2 systems."""
        L, replicas, J, h = setup_data['L'], setup_data['replicas'], setup_data['J'], setup_data['h']
        spins = setup_data['spins_ising']

        new_system = IsingSystem(
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=J,
            external_field=h,
            initial_spin_state=spins
        )

        with pytest.raises(TypeError, match="only valid for Z2GaugeSystem"):
            WilsonLoop(new_system, loop_size=1)

        with pytest.raises(TypeError, match="only valid for Z2GaugeSystem"):
            Plaquette(new_system)

    def test_gauge_placeholder(self):
        """Ensure Z2GaugeSystem placeholders raise NotImplementedError."""
        # Z2GaugeSystem might need init args even if placeholder?
        # Let's check constructor signature of Z2GaugeSystem.
        # It accepts *args, **kwargs and calls super which might fail if not provided.
        # BUT Z2Gauge placeholder __init__ is:
        # def __init__(self, *args, **kwargs):
        #    pass
        # So it doesn't call super().__init__. It defines nothing.
        # So instantiation Z2GaugeSystem() should work.

        system = Z2GaugeSystem()

        wl = WilsonLoop(system)
        with pytest.raises(NotImplementedError):
            wl.compute()

        pl = Plaquette(system)
        with pytest.raises(NotImplementedError):
            pl.compute()
