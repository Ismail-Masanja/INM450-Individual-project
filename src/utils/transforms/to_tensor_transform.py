import torch
from collections import namedtuple


__all__ = ['ToTensorTransform']


class ToTensorTransform:
    """Transform class to convert a namedtuple into a tensor.

    This class takes a network traffic data sample represented as a namedtuple and converts it
    into a PyTorch tensor. The fields of the namedtuple are automatically ordered alphabetically
    before conversion to ensure consistency in the tensor representation. Values are converted to
    float32 to enable computation at tensor cores if available, and values are clamped within the
    float32 range to prevent overflow or underflow issues.

    The conversion process involves sorting the fields of the namedtuple alphabetically, extracting
    their values, converting these values to float32, and then creating a tensor from these values.
    """

    def __call__(self, traffic: namedtuple) -> torch.Tensor:
        """Convert a namedtuple representing a traffic data sample into a tensor.

        Args:
            traffic (namedtuple): A namedtuple representing a single sample of network
                                  traffic data, containing various fields that will be
                                  converted into tensor form.

        Returns:
            torch.Tensor: A tensor representation of the traffic data sample, with values
                          clamped to the float32 range and fields ordered alphabetically.
        """
        # Convert namedtuple to dictionary for manipulation
        traffic_dict = traffic._asdict()

        # Sort dictionary by keys (field names) to ensure alphabetical order
        sorted_traffic_dict = dict(sorted(traffic_dict.items()))

        # Extract values in sorted order to ensure consistency
        sorted_values = [float(x) for x in sorted_traffic_dict.values()]

        # Convert to tensor, using float32 for computational efficiency
        tensor = torch.tensor(sorted_values, dtype=torch.float32)

        # Clamp values to float32 range to avoid overflow/underflow
        max_float32 = torch.finfo(torch.float32).max
        min_float32 = torch.finfo(torch.float32).min
        tensor = torch.clamp(tensor, min=min_float32, max=max_float32)
                
        return tensor
