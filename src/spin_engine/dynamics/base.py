import abc
import tensorflow as tf
from typing import Optional
from ..models.base import BaseSpinSystem
from ..measurements.tracker import Tracker


class BaseDynamics(abc.ABC):
    """
    Abstract base class for all dynamics drivers.
    Manages the evolution of the SpinSystem.
    """

    @abc.abstractmethod
    def step(self, system: BaseSpinSystem, **kwargs) -> tf.Tensor:
        """
        Perform a single update step on the system.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sweep(
        self,
        system: BaseSpinSystem,
        n_steps: int = 100,
        tracker: Optional[Tracker] = None,
        **kwargs
    ):
        """
        Perform a simulation sweep (multiple steps).

        Args:
            system: The spin system to evolve.
            n_steps: Number of steps to simulate.
            tracker: Optional Tracker instance to record observables.
            **kwargs: Algorithm-specific parameters (e.g., beta, temperature).
        """
        raise NotImplementedError
