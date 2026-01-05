# type: ignore

import tensorflow as tf
from typing import Optional
from .base import Measurement
from spin_engine.models.z2_gauge import Z2GaugeSystem


class Plaquette(Measurement):
    """
    Computes the average plaquette value.
    Placeholder until Z2GaugeSystem is implemented.
    """

    def __init__(self, system):
        super().__init__(system)
        if not isinstance(system, Z2GaugeSystem):
            raise TypeError(
                "Plaquette measurement only valid for Z2GaugeSystem")

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None) -> tf.Tensor:
        raise NotImplementedError("Z2GaugeSystem is not implemented yet.")


class WilsonLoop(Measurement):
    """
    Computes Wilson loops of a given size.
    Placeholder until Z2GaugeSystem is implemented.
    """

    def __init__(self, system, loop_size: int = 1):
        super().__init__(system)
        if not isinstance(system, Z2GaugeSystem):
            raise TypeError(
                "WilsonLoop measurement only valid for Z2GaugeSystem")
        self.loop_size = loop_size

    def compute(self, spin_state: Optional[tf.Variable | tf.Tensor] = None) -> tf.Tensor:
        raise NotImplementedError("Z2GaugeSystem is not implemented yet.")
