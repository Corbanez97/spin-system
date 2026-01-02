import tensorflow as tf
from typing import Optional, Union, Callable, Tuple, List
import abc

class BaseSpinSystem(tf.Module, abc.ABC):
    """
    Abstract base class for all spin systems.
    Encapsulates the lattice state, shape, and common utilities.
    """
    def __init__(
        self,
        lattice_dim: int,
        lattice_length: int,
        lattice_replicas: int,
        initial_spin_state: Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]] = None,
    ):
        super().__init__()
        self.lattice_dim = lattice_dim
        self.lattice_length = lattice_length
        self.lattice_replicas = lattice_replicas
        
        # Derived properties
        self.shape = [lattice_length] * lattice_dim
        self.number_spins = tf.cast(lattice_length ** lattice_dim, tf.float32)

        # Initialize or validate spin state
        self.spin_state = self._initialize_or_validate_state(initial_spin_state)
    
    @abc.abstractmethod
    def initialize_state(self) -> tf.Tensor:
        """
        Generates the initial spin state configuration.
        Must be implemented by subclasses.
        Returns:
            tf.Tensor of shape (replicas, *shape, [components])
        """
        pass

    @abc.abstractmethod
    def compute_energy(self, spin_state: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Computes the energy of the system for each replica.
        Args:
            spin_state: Optional tensor to compute energy for. Uses self.spin_state if None.
        Returns:
            tf.Tensor of shape (replicas,) containing energy values.
        """
        pass

    def _initialize_or_validate_state(
        self, 
        initial_state: Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]]
    ) -> tf.Variable:
        """
        Helper to handle state initialization logic.
        """
        if initial_state is None:
            initial_value = self.initialize_state()
        elif callable(initial_state):
            initial_value = initial_state()
        else:
            initial_value = tf.convert_to_tensor(initial_state, dtype=tf.float32)
            
        return tf.Variable(initial_value, trainable=True, dtype=tf.float32, name="spin_state")

    # TODO: Review _validate_tensor_shape method
    def _validate_tensor_shape(
        self,
        tensor: Optional[tf.Tensor],
        expected_shape: tuple[int, ...],
        name: str,
        allow_none: bool = False,
        default: Optional[Union[tf.Tensor, Callable[[], tf.Tensor]]] = None,
    ) -> tf.Tensor:
        """
        Convert input to tf.Tensor and validate its shape.
        """
        if tensor is None:
            if allow_none:
                if callable(default):
                    return default()
                elif default is not None:
                    return default
                # else:
                #     return None
            else:
                raise ValueError(f"{name} cannot be None")

        tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
        if tensor.shape != expected_shape:
            # Check for compatibility (e.g. broadcasting) if strict equality fails?
            # For now strict check as in legacy_core
            raise ValueError(
                f"{name} must be shape {expected_shape}, got {tensor.shape}"
            )
        return tensor