import tensorflow as tf
from typing import cast

from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.models import IsingSystem
from spin_engine.dynamics import MetropolisHastings

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
print(ising_system.spin_state[0, ...])  # type: ignore

simulation = MetropolisHastings(ising_system)

print(simulation.current_energy[0])  # type: ignore

num_flips = cast(tf.Tensor, tf.constant(1))

proposed_spin_state, energy_delta = simulation.flip_spins(num_flips=num_flips)

print(proposed_spin_state[0, ...])  # type: ignore
print(energy_delta[0])  # type: ignore
