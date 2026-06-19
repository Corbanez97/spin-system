"""
Lightweight benchmark for Spin System simulations.

Measures wall-clock time for single sweep() calls with smaller parameters
to get quick, actionable data about per-step costs and scaling.
"""

import time
import numpy as np
import tensorflow as tf
from typing import cast

from spin_engine.models import IsingSystem, EdwardsAndersonSystem
from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.interactions.standard import BinaryRandomInteraction
from spin_engine.dynamics import MetropolisHastings
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.scalars import Energy, Magnetization
from spin_engine.measurements.correlations import OverlapDistribution


def benchmark_sweep(label, system, measurements, sweep_length, granularity=100, beta=1.0):
    """Run two sweeps: first includes tf.function tracing, second is cached."""
    num_flips = cast(tf.Tensor, tf.constant(1))
    sim = MetropolisHastings(system)
    tracker = Tracker(measurements=measurements, granularity=granularity)

    # First sweep (includes tracing)
    t0 = time.perf_counter()
    sim.sweep(
        tracker=tracker,
        beta=tf.constant(beta, dtype=tf.float32),
        num_disturbances=num_flips,
        sweep_length=sweep_length,
    )
    t_trace = time.perf_counter() - t0

    # Second sweep (cached graph)
    t1 = time.perf_counter()
    sim.sweep(
        tracker=tracker,
        beta=tf.constant(beta, dtype=tf.float32),
        num_disturbances=num_flips,
        sweep_length=sweep_length,
    )
    t_run = time.perf_counter() - t1

    steps_per_sec = sweep_length / t_run
    print(f"  {label:<30s} | steps={sweep_length:>8d} | "
          f"trace={t_trace:>7.2f}s | cached={t_run:>7.2f}s | "
          f"{steps_per_sec:>8.0f} steps/s")

    return {
        "label": label,
        "sweep_length": sweep_length,
        "t_trace": t_trace,
        "t_run": t_run,
        "steps_per_sec": steps_per_sec,
    }


def main():
    print("=" * 90)
    print("SPIN SYSTEM BENCHMARK (Lite)")
    print(f"TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU: {gpus[0].name if gpus else 'None (CPU only)'}")
    print("=" * 90)

    replicas = 64
    granularity = 100
    # Use a fixed small number of sweeps for comparison
    SWEEPS = 10

    all_results = []

    # ---- 2D Ising ----
    print(f"\n[1] 2D Ising Model (replicas={replicas}, sweeps={SWEEPS})")
    print("-" * 90)
    for L in [8, 16, 32]:
        N = L * L
        sweep_length = SWEEPS * N
        interaction_matrix = PeriodicNearestNeighborInteraction().generate(2, L)
        system = IsingSystem(
            lattice_dim=2, lattice_length=L, lattice_replicas=replicas,
            interaction_matrix=interaction_matrix, initial_magnetization=1.0,
        )
        r = benchmark_sweep(
            f"Ising 2D L={L} N={N}",
            system, [Energy(system), Magnetization(system)],
            sweep_length, granularity,
        )
        r["model"] = "Ising 2D"
        r["L"] = L
        r["N"] = N
        r["D"] = 2
        all_results.append(r)

    # ---- 3D EA ----
    print(f"\n[2] 3D Edwards-Anderson Model (replicas={replicas}, sweeps={SWEEPS})")
    print("-" * 90)
    for L in [4, 6, 8]:
        N = L ** 3
        sweep_length = SWEEPS * N
        nn_mask = PeriodicNearestNeighborInteraction().generate(3, L)
        random_J = BinaryRandomInteraction(J=1.0, seed=42).generate(3, L)
        interaction_matrix = nn_mask * random_J
        system = EdwardsAndersonSystem(
            lattice_length=L, lattice_dim=3, lattice_replicas=replicas,
            interaction_matrix=interaction_matrix, initial_magnetization=0.0,
        )
        r = benchmark_sweep(
            f"EA 3D L={L} N={N}",
            system, [Energy(system), OverlapDistribution(system)],
            sweep_length, granularity,
        )
        r["model"] = "EA 3D"
        r["L"] = L
        r["N"] = N
        r["D"] = 3
        all_results.append(r)

    # ---- Measurement overhead (EA 3D L=4) ----
    print(f"\n[3] Measurement Overhead Comparison (EA 3D L=4, sweeps={SWEEPS})")
    print("-" * 90)
    L, N = 4, 64
    sweep_length = SWEEPS * N
    nn_mask = PeriodicNearestNeighborInteraction().generate(3, L)
    random_J = BinaryRandomInteraction(J=1.0, seed=42).generate(3, L)
    interaction_matrix = nn_mask * random_J

    for label, make_meas in [
        ("Energy only", lambda s: [Energy(s)]),
        ("Energy+Mag", lambda s: [Energy(s), Magnetization(s)]),
        ("Energy+Overlap", lambda s: [Energy(s), OverlapDistribution(s)]),
        ("All three", lambda s: [Energy(s), Magnetization(s), OverlapDistribution(s)]),
    ]:
        system = EdwardsAndersonSystem(
            lattice_length=L, lattice_dim=3, lattice_replicas=replicas,
            interaction_matrix=interaction_matrix, initial_magnetization=0.0,
        )
        r = benchmark_sweep(
            f"EA3D L=4 [{label}]",
            system, make_meas(system), sweep_length, granularity,
        )
        all_results.append(r)

    # ---- Granularity impact ----
    print(f"\n[4] Granularity Impact (EA 3D L=4, sweeps={SWEEPS})")
    print("-" * 90)
    for gran in [50, 100, 500, 1000]:
        system = EdwardsAndersonSystem(
            lattice_length=L, lattice_dim=3, lattice_replicas=replicas,
            interaction_matrix=interaction_matrix, initial_magnetization=0.0,
        )
        num_recordings = sweep_length // gran
        r = benchmark_sweep(
            f"EA3D L=4 gran={gran} ({num_recordings} recs)",
            system, [Energy(system), OverlapDistribution(system)],
            sweep_length, gran,
        )
        all_results.append(r)

    # ---- Projections ----
    print("\n" + "=" * 90)
    print("PROJECTED RUNTIMES FOR EXAMPLE SCRIPTS")
    print("=" * 90)

    # Build a lookup of cached steps/sec by (model, N)
    rate_by = {}
    for r in all_results:
        if "N" in r:
            rate_by[(r.get("model", ""), r["N"])] = r["steps_per_sec"]

    scripts = [
        {
            "name": "examples/ea_observables.py",
            "configs": [
                {"model": "EA 3D", "L": 4, "N": 64, "sweeps": 4000, "betas": 5},
                {"model": "EA 3D", "L": 6, "N": 216, "sweeps": 3000, "betas": 5},
            ]
        },
        {
            "name": "examples/ea_glass.py",
            "configs": [
                {"model": "EA 3D", "L": 4, "N": 64, "sweeps": 5000, "betas": 25},
                {"model": "EA 3D", "L": 8, "N": 512, "sweeps": 5000, "betas": 25},
            ]
        },
        {
            "name": "examples/ising.py",
            "configs": [
                {"model": "Ising 2D", "L": 8, "N": 64, "sweeps": 2344, "betas": 25},
                {"model": "Ising 2D", "L": 16, "N": 256, "sweeps": 1172, "betas": 25},
                {"model": "Ising 2D", "L": 32, "N": 1024, "sweeps": 586, "betas": 25},
            ]
        },
    ]

    for script in scripts:
        print(f"\n  --- {script['name']} ---")
        total = 0
        for cfg in script["configs"]:
            key = (cfg["model"], cfg["N"])
            sweep_length = cfg["sweeps"] * cfg["N"]
            if key in rate_by:
                rate = rate_by[key]
                t_per_beta = sweep_length / rate
                t_total = t_per_beta * cfg["betas"]
                total += t_total
                print(f"    L={cfg['L']:>2d} N={cfg['N']:>5d}: "
                      f"{cfg['betas']} betas × {sweep_length:>9d} steps "
                      f"@ {rate:.0f} steps/s ≈ {t_total/60:.1f} min")
            else:
                print(f"    L={cfg['L']:>2d} N={cfg['N']:>5d}: (no benchmark data)")
        if total > 0:
            print(f"    TOTAL ≈ {total/60:.1f} min ({total/3600:.2f} hours)")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
