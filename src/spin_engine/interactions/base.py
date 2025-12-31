import abc
import numpy as np


class Interaction(abc.ABC):
    """
    Abstract base class for all interaction types.
    """
    @abc.abstractmethod
    def generate(self, D: int, L: int) -> np.ndarray:
        """
        Generates the interaction matrix/tensor.

        Args:
           D: Dimension of the lattice
           L: Length of the lattice side

        Returns:
            np.ndarray: Interaction tensor of shape (L,)*D*2 or matrix (N, N) depending on usage,
                        but consistently (L,)*D*2 for the provided examples.
        """
        pass
