"""
Benchmarking script for Spin System simulations.

Measures wall-clock time for single sweep() calls across different models,
lattice sizes, and configurations. Outputs a summary table.
"""

import time
import sys
import os
import numpy as np
import tensorflow as tf
from typing import cast

from spin_engine.models import IsingSystem, EdwardsAndersonSystem, SherringtonKirkpatrickSystem
from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.interactions.standard import BinaryRandomInteraction
from spin_engine.dynamics import MetropolisHastings
from spin_engine.dynamics.tracker import Tracker
from spin_engine.measurements.scalars import Energy, Magnetization
from spin_engine.measurements.correlations import OverlapDistribution


def benchmark_single_sweep(model_name, system, measurements, sweep_length, granularity=100, beta=1.0):
    """Run a single sweep and return (wall_time_seconds, steps_per_second)."""
    num_flips = cast(tf.Tensor, tf.constant(1))

    sim = MetropolisHastings(system)
    tracker = Tracker(measurements=measurements, granularity=granularity)

    # Warm-up: first call includes tf.function tracing
    t0 = time.perf_counter()
    sim.sweep(
        tracker=tracker,
        beta=tf.constant(beta, dtype=tf.float32),
        num_disturbances=num_flips,
        sweep_length=sweep_length,
    )
    t_trace = time.perf_counter() - t0

    # Second call: no retracing (same signature)
    t1 = time.perf_counter()
    sim.sweep(
        tracker=tracker,
        beta=tf.constant(beta, dtype=tf.float32),
        num_disturbances=num_flips,
        sweep_length=sweep_length,
    )
    t_run = time.perf_counter() - t1

    return t_trace, t_run


def benchmark_ising_2d():
    """Benchmark 2D Ising model at various lattice sizes."""
    results = []
    replicas = 64
    granularity = 100

    for L in [8, 16, 32, 64]:
        N = L * L
        sweeps = 2000
        sweep_length = sweeps * N

        interaction_matrix = PeriodicNearestNeighborInteraction().generate(2, L)
        system = IsingSystem(
            lattice_dim=2,
            lattice_length=L,
            lattice_replicas=replicas,
            interaction_matrix=interaction_matrix,
            initial_magnetization=1.0,
        )
        measurements = [Energy(system), Magnetization(system)]

        t_trace, t_run = benchmark_single_sweep(
            "Ising2D", system, measurements, sweep_length, granularity
        )

        results.append({
            "model": "Ising 2D",
            "L": L,
            "N": N,
            "replicas": replicas,
            "sweeps": sweeps,
            "total_steps": sweep_length,
            "measurements": "Energy+Mag",
            "t_trace_s": t_trace,
            "t_run_s": t_run,
            "steps_per_sec_traced": sweep_length / t_trace,
            "steps_per_sec_cached": sweep_length / t_run,
        })
        print(f"  Ising 2D L={L:>3d} | N={N:>5d} | steps={sweep_length:>9d} | "
              f"trace={t_trace:>8.2f}s | run={t_run:>8.2f}s | "
              f"cached rate={sweep_length/t_run:>10.0f} steps/s")

    return results


def benchmark_ea_3d():
    """Benchmark 3D Edwards-Anderson model at various lattice sizes."""
    results = []
    replicas = 64
    granularity = 100

    for L in [4, 6, 8]:
        N = L ** 3
        sweeps = 2000
        sweep_length = sweeps * N

        nn_mask = PeriodicNearestNeighborInteraction().generate(3, L)
        random_J = BinaryRandomInteraction(J=1.0, seed=42).generate(3, L)
        interaction_matrix = nn_mask * random_J

        system = EdwardsAndersonSystem(
            lattice_length=L,
            lattice_dim=3,
            lattice_replicas=replicas,
            interaction_matrix=interaction_matrix,
            initial_magnetization=0.0,
        )
        measurements = [Energy(system), OverlapDistribution(system)]

        t_trace, t_run = benchmark_single_sweep(
            "EA3D", system, measurements, sweep_length, granularity
        )

        results.append({
            "model": "EA 3D",
            "L": L,
            "N": N,
            "replicas": replicas,
            "sweeps": sweeps,
            "total_steps": sweep_length,
            "measurements": "Energy+Overlap",
            "t_trace_s": t_trace,
            "t_run_s": t_run,
            "steps_per_sec_traced": sweep_length / t_trace,
            "steps_per_sec_cached": sweep_length / t_run,
        })
        print(f"  EA 3D    L={L:>3d} | N={N:>5d} | steps={sweep_length:>9d} | "
              f"trace={t_trace:>8.2f}s | run={t_run:>8.2f}s | "
              f"cached rate={sweep_length/t_run:>10.0f} steps/s")

    return results


def benchmark_ea_3d_measurements_comparison():
    """Compare sweep time with different measurement configurations."""
    results = []
    L = 4
    N = L ** 3
    replicas = 64
    granularity = 100
    sweeps = 2000
    sweep_length = sweeps * N

    nn_mask = PeriodicNearestNeighborInteraction().generate(3, L)
    random_J = BinaryRandomInteraction(J=1.0, seed=42).generate(3, L)
    interaction_matrix = nn_mask * random_J

    configs = [
        ("Energy only", lambda sys: [Energy(sys)]),
        ("Energy+Mag", lambda sys: [Energy(sys), Magnetization(sys)]),
        ("Energy+Overlap", lambda sys: [Energy(sys), OverlapDistribution(sys)]),
        ("Energy+Mag+Overlap", lambda sys: [Energy(sys), Magnetization(sys), OverlapDistribution(sys)]),
    ]

    for label, make_measurements in configs:
        system = EdwardsAndersonSystem(
            lattice_length=L,
            lattice_dim=3,
            lattice_replicas=replicas,
            interaction_matrix=interaction_matrix,
            initial_magnetization=0.0,
        )
        measurements = make_measurements(system)

        t_trace, t_run = benchmark_single_sweep(
            f"EA3D-{label}", system, measurements, sweep_length, granularity
        )

        results.append({
            "config": label,
            "t_trace_s": t_trace,
            "t_run_s": t_run,
        })
        print(f"  EA 3D L=4 [{label:>20s}] | trace={t_trace:>8.2f}s | run={t_run:>8.2f}s")

    return results


def benchmark_granularity_impact():
    """Measure how tracking granularity affects performance."""
    results = []
    L = 4
    N = L ** 3
    replicas = 64
    sweeps = 2000
    sweep_length = sweeps * N

    nn_mask = PeriodicNearestNeighborInteraction().generate(3, L)
    random_J = BinaryRandomInteraction(J=1.0, seed=42).generate(3, L)
    interaction_matrix = nn_mask * random_J

    for granularity in [50, 100, 500, 1000]:
        system = EdwardsAndersonSystem(
            lattice_length=L,
            lattice_dim=3,
            lattice_replicas=replicas,
            interaction_matrix=interaction_matrix,
            initial_magnetization=0.0,
        )
        measurements = [Energy(system), OverlapDistribution(system)]

        t_trace, t_run = benchmark_single_sweep(
            f"EA3D-gran{granularity}", system, measurements,
            sweep_length, granularity
        )

        num_recordings = sweep_length // granularity
        results.append({
            "granularity": granularity,
            "num_recordings": num_recordings,
            "t_trace_s": t_trace,
            "t_run_s": t_run,
        })
        print(f"  EA 3D L=4 gran={granularity:>5d} | recordings={num_recordings:>6d} | "
              f"trace={t_trace:>8.2f}s | run={t_run:>8.2f}s")

    return results


def estimate_full_script_times(ising_results, ea_results):
    """Use benchmark data to estimate total runtimes for example scripts."""
    print("\n" + "="*80)
    print("ESTIMATED FULL SCRIPT RUNTIMES")
    print("="*80)

    # --- ea_observables.py ---
    # L=[4,6], 5 betas each, sweeps scaled by L
    print("\n--- examples/ea_observables.py ---")
    ea_configs = [
        {"L": 4, "sweeps": 4000, "betas": 5},
        {"L": 6, "sweeps": 3000, "betas": 5},
    ]
    total_ea_obs = 0
    for cfg in ea_configs:
        L = cfg["L"]
        N = L ** 3
        sweep_length = cfg["sweeps"] * N
        # Find matching benchmark
        matching = [r for r in ea_results if r["L"] == L]
        if matching:
            r = matching[0]
            # Scale from benchmark sweep_length to actual sweep_length
            scale = sweep_length / r["total_steps"]
            # First beta includes tracing, rest use cached graph
            t_first = r["t_trace_s"] * scale
            t_rest = r["t_run_s"] * scale * (cfg["betas"] - 1)
            t_total = t_first + t_rest
            total_ea_obs += t_total
            print(f"  L={L}: {cfg['betas']} betas × {sweep_length} steps "
                  f"≈ {t_total/60:.1f} min (trace: {t_first/60:.1f}m + cached: {t_rest/60:.1f}m)")
    print(f"  TOTAL estimated: {total_ea_obs/60:.1f} min ({total_ea_obs/3600:.1f} hours)")

    # --- ea_glass.py ---
    print("\n--- examples/ea_glass.py ---")
    ea_glass_configs = [
        {"L": 4, "sweeps": 5000, "betas": 25},
        {"L": 8, "sweeps": 5000, "betas": 25},
    ]
    total_ea_glass = 0
    for cfg in ea_glass_configs:
        L = cfg["L"]
        N = L ** 3
        sweep_length = cfg["sweeps"] * N
        matching = [r for r in ea_results if r["L"] == L]
        if matching:
            r = matching[0]
            scale = sweep_length / r["total_steps"]
            t_first = r["t_trace_s"] * scale
            t_rest = r["t_run_s"] * scale * (cfg["betas"] - 1)
            t_total = t_first + t_rest
            total_ea_glass += t_total
            print(f"  L={L}: {cfg['betas']} betas × {sweep_length} steps "
                  f"≈ {t_total/60:.1f} min (trace: {t_first/60:.1f}m + cached: {t_rest/60:.1f}m)")
    # L=16 and L=32 don't have benchmarks, estimate from scaling
    for L in [16, 32]:
        N = L ** 3
        sweep_length = 5000 * N
        # Extrapolate from L=8 scaling (roughly linear in N for matmul-dominated cost)
        r8 = [r for r in ea_results if r["L"] == 8]
        if r8:
            r = r8[0]
            # Cost scales roughly as N^2 (matmul) * sweep_length
            scale_n = (N / r["N"]) ** 2
            scale_len = sweep_length / r["total_steps"]
            t_est = r["t_run_s"] * scale_n * scale_len * 25
            total_ea_glass += t_est
            print(f"  L={L}: 25 betas × {sweep_length} steps "
                  f"≈ {t_est/60:.1f} min (EXTRAPOLATED — N²·steps scaling)")
    print(f"  TOTAL estimated: {total_ea_glass/60:.1f} min ({total_ea_glass/3600:.1f} hours)")

    # --- ising.py ---
    print("\n--- examples/ising.py ---")
    ising_configs = [
        {"L": 8, "sweep_length": 150000, "betas": 25},
        {"L": 16, "sweep_length": 300000, "betas": 25},
        {"L": 32, "sweep_length": 600000, "betas": 25},
    ]
    total_ising = 0
    for cfg in ising_configs:
        L = cfg["L"]
        matching = [r for r in ising_results if r["L"] == L]
        if matching:
            r = matching[0]
            scale = cfg["sweep_length"] / r["total_steps"]
            t_first = r["t_trace_s"] * scale
            t_rest = r["t_run_s"] * scale * (cfg["betas"] - 1)
            t_total = t_first + t_rest
            total_ising += t_total
            print(f"  L={L}: {cfg['betas']} betas × {cfg['sweep_length']} steps "
                  f"≈ {t_total/60:.1f} min")
    print(f"  TOTAL estimated: {total_ising/60:.1f} min ({total_ising/3600:.1f} hours)")


def main():
    print("="*80)
    print("SPIN SYSTEM BENCHMARK")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print("="*80)

    print("\n[1/4] Benchmarking 2D Ising Model")
    print("-" * 60)
    ising_results = benchmark_ising_2d()

    print("\n[2/4] Benchmarking 3D Edwards-Anderson Model")
    print("-" * 60)
    ea_results = benchmark_ea_3d()

    print("\n[3/4] Benchmarking Measurement Overhead (EA 3D L=4)")
    print("-" * 60)
    meas_results = benchmark_ea_3d_measurements_comparison()

    print("\n[4/4] Benchmarking Granularity Impact (EA 3D L=4)")
    print("-" * 60)
    gran_results = benchmark_granularity_impact()

    # Estimate full script runtimes
    estimate_full_script_times(ising_results, ea_results)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
