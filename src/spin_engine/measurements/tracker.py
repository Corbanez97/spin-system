import tensorflow as tf
from typing import List, Dict, Any, Optional
from ..models.base import BaseSpinSystem
from .base import Measurement


class Tracker:
    """
    Tracks observables during a simulation sweep using TensorArray.
    Designed to work within tf.function and tf.while_loop.
    """

    def __init__(
        self,
        measurements: List[Measurement],
        sweep_length: int,
        granularity: int = 1
    ):
        """
        Args:
            measurements: List of Measurement objects to track.
            sweep_length: Total number of steps in the sweep.
            granularity: Record measurements every N steps.
        """
        self.measurements = measurements
        self.granularity = granularity
        self.sweep_length = sweep_length
        self.size = (sweep_length // granularity) + 1

    def init(self) -> Dict[str, tf.TensorArray]:
        """
        Initialize the TensorArrays for tracking.
        Returns:
            Dict[str, tf.TensorArray]: Initial state of tracker history.
        """
        history = {}
        for measure in self.measurements:
            name = measure.__class__.__name__
            history[name] = tf.TensorArray(
                dtype=tf.float32,
                size=self.size,
                element_shape=None,
                clear_after_read=False
            )
        return history

    def record(
        self,
        step: tf.Tensor,
        system: BaseSpinSystem,
        history: Dict[str, tf.TensorArray]
    ) -> Dict[str, tf.TensorArray]:
        """
        Records observables if the current step matches the granularity.

        Args:
            step: Current simulation step (tensor).
            system: The spin system state to measure.
            history: Current dictionary of TensorArrays.

        Returns:
            Updated dictionary of TensorArrays.
        """

        should_record = (step % self.granularity == 0)

        def _write():
            idx = step // self.granularity
            new_history = {}
            for measure in self.measurements:
                name = measure.__class__.__name__
                val = measure.compute(system.spin_state)
                # Note: creating a new dict for the updated TAs
                new_history[name] = history[name].write(
                    tf.cast(idx, tf.int32), val)
            return new_history

        def _skip():
            return history

        return tf.cond(should_record, _write, _skip)

    def finalize(self, history: Dict[str, tf.TensorArray]) -> Dict[str, tf.Tensor]:
        """
        Finalize the tracking by stacking the TensorArrays.

        Args:
            history: The final dictionary of TensorArrays.

        Returns:
            Dict[str, tf.Tensor]: Stacked results.
        """
        results = {}
        for name, ta in history.items():
            results[name] = ta.stack()
        return results
