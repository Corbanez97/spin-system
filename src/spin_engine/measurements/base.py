from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem


class Measurement(tf.Module, ABC):
    """
    Abstract base class for all measurements.

    A Measurement is an observer that can compute a value (scalar or tensor)
    given a Spin System and its state.
    """

    def __init__(self, system: 'BaseSpinSystem', name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.system = system

    @abstractmethod
    def compute(self, spin_state: Optional[tf.Tensor] = None) -> Any:
        """
        Compute the measurement.

        Args:
            spin_state: Optional tensor representing the state to measure.
                        If None, uses self.system.spin_state.

        Returns:
            The computed measurement value.
        """
        pass

    def __call__(self, spin_state: Optional[tf.Tensor] = None) -> Any:
        return self.compute(spin_state)
