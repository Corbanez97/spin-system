import abc
import tensorflow as tf

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem
    from spin_engine.dynamics.tracker import Tracker


class Dynamics(abc.ABC):
    """
    Abstract base class for all dynamics

    The Dynamics dictates how the spin state evolves over time.
    """

    def __init__(self, system: 'BaseSpinSystem', ) -> None:
        self.system = system

    @abc.abstractmethod
    # @tf.function
    def step(
        self,
        beta: float,
        num_disturbances: tf.Tensor,
        theta_max: Optional[tf.Tensor] = None
    ) -> 'BaseSpinSystem':
        """
        How a step is taken inside our simulation. This method should be called in the main loop of the simulation.
        """
        pass

    @abc.abstractmethod
    @tf.function
    def sweep(
        self,
        tracker: 'Tracker',
        beta: float,
        num_disturbances: tf.Tensor,
        theta_max: Optional[tf.Tensor] = None,
        sweep_length: Optional[int] = None,
    ) -> 'BaseSpinSystem':
        """
        The orchestrator of multiple steps of the simulation.
        """
        pass
