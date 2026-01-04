from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spin_engine.models.base import BaseSpinSystem


class Measurement(ABC):
    """
    Abstract base class for all measurements.

    A Measurement is an observer that can compute a value (scalar or tensor)
    given a Spin System and its state.
    """

    def __init__(self, system: Optional['BaseSpinSystem'] = None) -> None:
        self.system = system

    @abstractmethod
    def compute(self, spin_state: Optional[tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> Any:
        """
        Compute the measurement.

        Args:
            spin_state: Optional tensor representing the state to measure.
                        If None, uses self.system.spin_state.
            system: Optional BaseSpinSystem instance. If None, uses self.system.

        Returns:
            The computed measurement value.
        """
        pass

    def __call__(self, spin_state: Optional[tf.Tensor] = None, system: Optional['BaseSpinSystem'] = None) -> Any:
        return self.compute(spin_state, system)
