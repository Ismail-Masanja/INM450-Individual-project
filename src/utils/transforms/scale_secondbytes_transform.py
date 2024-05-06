from collections import namedtuple
import torch


__all__ = ['ScaleSecondBytesTransform']


class ScaleSecondBytesTransform:
    """Transform class to scale 'SRC_TO_DST_SECOND_BYTES' and 'DST_TO_SRC_SECOND_BYTES' fields.

    This class takes a network traffic data sample represented as a namedtuple and scales
    the 'SRC_TO_DST_SECOND_BYTES' and 'DST_TO_SRC_SECOND_BYTES' fields to a range of [-1, 1]
    using an exponential scaling method similar to a sigmoid function.

    The scaling is controlled by an 'extreme_value' parameter which determines the
    scaling's sensitivity to changes in the input values.

    Attributes:
        extreme_value (float): A value that defines the extreme value for the scaling
                               function. Larger values make the scaling less sensitive
                               to input changes.
    """

    def __init__(self, extreme_value: float = 1e10):
        """Initialize the scaling transform with an optional extreme value parameter.

        Args:
            extreme_value (float): The extreme value used for scaling calculations.
        """
        self.extreme_value = extreme_value

    def exponential_scale(self, value: torch.Tensor) -> float:
        """Applies an exponential scaling function to a given value.

        Args:
            value (torch.Tensor): The input value to scale.

        Returns:
            float: The scaled value.
        """
        # Calculate scaling factor based on the extreme value
        k = 1 / self.extreme_value
        # Apply the sigmoid-like scaling formula
        scaled_value = 2 / (1 + torch.exp(-k * value)) - 1
        return scaled_value.item()

    def __call__(self, traffic: namedtuple) -> namedtuple:
        """Apply the exponential scaling transform to the specified fields of a traffic sample.

        Args:
            traffic (namedtuple): A namedtuple representing a single sample of network
                                  traffic data, possibly containing second byte fields.

        Returns:
            namedtuple: A new namedtuple instance with the second byte fields scaled
                        to the range [-1, 1].
        """
        # Convert namedtuple to dictionary for manipulation
        traffic_dict = traffic._asdict()

        # Fields to scale exponentially
        scale_fields = ['SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES']

        for field in scale_fields:
            if field in traffic_dict and traffic_dict[field] is not None:
                # Apply exponential scaling
                traffic_dict[field] = self.exponential_scale(
                    torch.tensor(float(traffic_dict[field])))

        # Dynamically recreate namedtuple with scaled fields
        Traffic = namedtuple('Traffic', traffic_dict.keys())
        return Traffic(**traffic_dict)
