from collections import namedtuple


__all__ = ['NormalizeProtocolFieldTransform']


class NormalizeProtocolFieldTransform:
    """Transform class to normalize the 'PROTOCOL' field in network traffic data.

    This class takes a network traffic data sample represented as a namedtuple and normalizes
    the 'PROTOCOL' field value to a range of [-1, 1], based on the assumption that protocol
    numbers range from 0 to 255. This normalization helps in preparing the data for machine
    learning models by ensuring that protocol numbers are on a similar scale to other features
    in the dataset.

    Attributes:
        max_protocol_number (int): The maximum value for the protocol number used for normalization.
    """

    def __init__(self, max_protocol_number: int = 255):
        """Initialize the normalization transform with a maximum protocol number.

        Args:
            max_protocol_number (int): The maximum value for the protocol number,
                                       defaulting to 255.
        """
        self.max_protocol_number = max_protocol_number

    def __call__(self, traffic: namedtuple) -> namedtuple:
        """Apply normalization to the 'PROTOCOL' field of a traffic data sample.

        Args:
            traffic (namedtuple): A namedtuple representing a single sample of network
                                  traffic data, containing a 'PROTOCOL' field among
                                  potentially others.

        Returns:
            namedtuple: A new namedtuple instance with the 'PROTOCOL' field normalized
                        to the range [-1, 1].
        """
        # Convert the namedtuple to a dictionary for manipulation
        traffic_dict = traffic._asdict()

        # Normalize 'PROTOCOL' field
        for field in ['PROTOCOL']:
            if field in traffic_dict:
                # Calculate the normalized value
                normalized_value = (
                    float(traffic_dict[field]) / self.max_protocol_number) * 2 - 1
                traffic_dict[field] = normalized_value

        # Dynamically recreate namedtuple with normalized fields
        Traffic = namedtuple('Traffic', traffic_dict.keys())
        return Traffic(**traffic_dict)
