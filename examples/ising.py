# type: ignore

import tensorflow as tf
from typing import cast

from spin_engine.interactions import PeriodicNearestNeighborInteraction
from spin_engine.models import IsingSystem
from spin_engine.dynamics import MetropolisHastings

lattice_dim = 2
lattice_length = 32
lattice_replicas = 6

interaction_matrix = PeriodicNearestNeighborInteraction().generate(
    lattice_dim, lattice_length)

ising_system = IsingSystem(
    lattice_dim=lattice_dim,
    lattice_length=lattice_length,
    lattice_replicas=lattice_replicas,
    interaction_matrix=interaction_matrix
)

simulation = MetropolisHastings(ising_system)

num_flips = cast(tf.Tensor, tf.constant(1))

proposed_system = simulation.step(beta=0.5, numb_disturbances=num_flips)
