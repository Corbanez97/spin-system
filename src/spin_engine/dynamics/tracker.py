import tensorflow as tf
from typing import Sequence, Dict, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from spin_engine.measurements.base import Measurement
    from spin_engine.models.base import BaseSpinSystem


class Tracker(tf.Module):
    def __init__(self, measurements: Sequence['Measurement'], granularity: int = 1):
        super().__init__()
        self.measurements = measurements
        self.granularity = granularity
        self.history = {}

    def init_run(self, sweep_length: tf.Tensor) -> Dict[str, tf.TensorArray]:
        """
        Initialize the tracking arrays for the run.
        """
        num_measurements = tf.cast(
            tf.divide(sweep_length, self.granularity), tf.int32)
        # Size = num_measurements + 1 (for initial state or inclusive boundary if handled)
        size = tf.add(num_measurements, 1)
        arrays = {}
        for measurement in self.measurements:
            name = getattr(measurement, 'name', measurement.__class__.__name__)
            # Use dynamic size if needed, but fixed size is better for performance if known.
            # clear_after_read=False is needed if we stack at end.
            arrays[name] = tf.TensorArray(
                dtype=tf.float32, size=size, clear_after_read=True)

        return arrays

    def track(
        self,
        step: tf.Tensor,
        system: 'BaseSpinSystem',
        tracking_arrays: Dict[str, tf.TensorArray]
    ) -> Dict[str, tf.TensorArray]:
        """
        Track measurements if step is a multiple of granularity.
        To be called inside tf.while_loop body.
        """

        step_int = tf.cast(step, tf.int32)

        def write_measurements(arrays):
            index = tf.math.floordiv(step_int, self.granularity)

            new_arrays = {}
            for measurement in self.measurements:
                name = getattr(measurement, 'name',
                               measurement.__class__.__name__)
                val = measurement.compute(system.spin_state, system=system)
                new_arrays[name] = arrays[name].write(index, val)
            return new_arrays

        def no_op(arrays):
            return arrays

        condition = tf.equal(tf.math.floormod(step_int, self.granularity), 0)

        return cast(
            Dict[str, tf.TensorArray],
            tf.cond(condition, lambda: write_measurements(
                tracking_arrays), lambda: no_op(tracking_arrays))
        )

    def finalize(self, tracking_arrays: Dict[str, tf.TensorArray]):
        """
        Stack results and store in self.history.
        """
        for name, array in tracking_arrays.items():
            result = array.stack()
            if name not in self.history:
                self.history[name] = tf.Variable(
                    result, validate_shape=False,  trainable=False)
            else:
                self.history[name].assign(result)
