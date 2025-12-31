import numpy as np
from .base import Interaction


class DecayingInteraction(Interaction):
    def __init__(self, J0: float = 10, alpha: float = 1):
        self.J0 = J0
        self.alpha = alpha

    def generate(self, D: int, L: int) -> np.ndarray:
        coords = np.array(np.meshgrid(
            *[np.arange(L)]*D, indexing='ij')).reshape(D, -1).T

        # Pairwise Euclidean distances
        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        J_flat = self.J0 * np.exp(-self.alpha * distances)
        np.fill_diagonal(J_flat, 0)

        # Vectorized reshape
        tensor_shape = (L,)*D*2
        J_tensor = J_flat.reshape(tensor_shape)

        return J_tensor


class PeriodicNearestNeighborInteraction(Interaction):
    """
    Vectorized nearest-neighbor coupling tensor with periodic boundaries.
    J[i1,...,iD,j1,...,jD] = 1 if periodic Manhattan distance = 1, else 0
    """

    def generate(self, D: int, L: int) -> np.ndarray:
        # Generate all coordinates: shape (N, D)
        coords = np.array(np.meshgrid(
            *[np.arange(L)]*D, indexing='ij')).reshape(D, -1).T
        N = coords.shape[0]

        # Compute pairwise differences with broadcasting
        diff = np.abs(coords[:, None, :] - coords[None, :, :])

        # Apply periodic boundary
        diff = np.minimum(diff, L - diff)

        # Manhattan distance
        manhattan_dist = diff.sum(axis=2)

        # Nearest neighbors mask
        nn_mask = (manhattan_dist == 1)

        # Create empty tensor and set neighbors
        J_tensor = np.zeros((L,)*D*2, dtype=np.float32)

        # Get indices where nn_mask is True
        idx_i, idx_j = np.nonzero(nn_mask)

        # Set values in the tensor
        for i, j in zip(idx_i, idx_j):
            J_tensor[tuple(coords[i]) + tuple(coords[j])] = 1.0

        return J_tensor


class CurieWeissInteraction(Interaction):
    def __init__(self, J0: float = 1.0):
        self.J0 = J0

    def generate(self, D: int, L: int) -> np.ndarray:
        N = L**D

        J_flat = (self.J0 / N) * (np.ones((N, N)) - np.eye(N))

        tensor_shape = (L,) * D * 2
        J_tensor = J_flat.reshape(tensor_shape)

        return J_tensor


class GaussianInteraction(Interaction):
    def __init__(self, mean: float = 0.0, std: float = 1.0, seed: int = None):
        self.mean = mean
        self.std = std
        self.seed = seed

    def generate(self, D: int, L: int) -> np.ndarray:
        if self.seed is not None:
            np.random.seed(self.seed)

        N = L**D
        J_flat = np.random.normal(self.mean, self.std, size=(N, N))

        J_flat = 0.5 * (J_flat + J_flat.T)

        np.fill_diagonal(J_flat, 0)

        tensor_shape = (L,) * D * 2
        J_tensor = J_flat.reshape(tensor_shape)

        return J_tensor
