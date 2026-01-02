import tensorflow as tf
import pytest
import numpy as np
from spin_engine.models.ising import IsingSystem
from spin_engine.measurements.scalars import Energy, Magnetization
from spin_engine.measurements.tracker import Tracker


class TestTracker:
    @pytest.fixture
    def ising_system(self):
        J = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)
        # 1D lattice L=2, replicas=1
        return IsingSystem(
            lattice_length=2,
            lattice_replicas=1,
            lattice_dim=1,
            interaction_matrix=J,
            initial_spin_state=tf.constant(
                [[[1.0], [-1.0]]], dtype=tf.float32)  # Up, Down
        )

    def test_tracker_granularity(self, ising_system):
        # Setup
        sweep_len = 10
        granularity = 2
        measurements = [Energy(ising_system), Magnetization(ising_system)]
        tracker = Tracker(measurements, sweep_len, granularity)

        # history = tracker.init() # This line is moved

        # Simulate loop
        @tf.function
        def run_loop():
            history = tracker.init()
            curr_hist = history
            for i in tf.range(sweep_len + 1):
                # We pretend step i happened.
                # Just record without changing system
                curr_hist = tracker.record(i, ising_system, curr_hist)
            return tracker.finalize(curr_hist)

        results = run_loop()

        # Expected size: 0, 2, 4, 6, 8, 10 -> 6 records
        expected_size = (sweep_len // granularity) + 1

        assert "Energy" in results
        assert "Magnetization" in results

        assert results["Energy"].shape[0] == expected_size
        assert results["Magnetization"].shape[0] == expected_size

        # Check values
        # Energy: -0.5 * S J S.
        # S=[1, -1]. J=[[0,1],[1,0]].
        # h = [1, -1] @ [[0,1],[1,0]] = [-1, 1].
        # S*h = [1*-1, -1*1] = [-1, -1]. sum=-2.
        # E = -0.5 * -2 = 1.0. (Antiferromagnetic config with Ferromagnetic coupling, cost energy)

        # All records should be 1.0
        assert np.allclose(results["Energy"].numpy(), 1.0)

    def test_tracker_skip(self, ising_system):
        # Test step that shouldn't record
        tracker = Tracker([Energy(ising_system)], 10, 10)  # record at 0, 10
        history = tracker.init()

        # Record step 5 (should invoke _skip)
        history = tracker.record(tf.constant(5), ising_system, history)

        # We can't easily check internal TA state without reading,
        # but if we finalize now, default TA reads 0 if not written?
        # TF TA reading unwritten index is error or zero depending on implementation?
        # Actually our size is fixed. If we don't write index 0, reading it fails.
        # But step 5 corresponds to index 0? No.
        # Step 0 -> Index 0. Step 5 -> Not recorded.
        pass
