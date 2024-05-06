from collections import namedtuple


__all__ = ['NormalizePortTransform']


class NormalizePortTransform:
    """A transform class for normalizing Layer 4 source and destination ports.

    This class takes a network traffic data sample represented as a namedtuple and normalizes
    the values of Layer 4 source and destination ports (L4_SRC_PORT and L4_DST_PORT) to a range
    of [-1, 1] based on the maximum possible port number (65535).

    After normalization, it returns a new namedtuple with the normalized port values.
    """

    def __call__(self, traffic: namedtuple) -> namedtuple:
        """Normalize the L4 source and destination port fields of a traffic data sample.

        Args:
            traffic (namedtuple): A namedtuple representing a single sample of network
                                  traffic data, containing L4 source and destination port
                                  fields among potentially others.

        Returns:
            namedtuple: A new namedtuple instance with the L4 source and destination port
                        fields normalized to the range [-1, 1].
        """
        # Convert the namedtuple to a dictionary for manipulation
        traffic_dict = traffic._asdict()

        # Normalize ports
        for port_field in ['L4_SRC_PORT', 'L4_DST_PORT']:
            if port_field in traffic_dict:
                # Apply normalization formula
                normalized_port = (float(traffic_dict[port_field]) / 65535.0) * 2 - 1
                traffic_dict[port_field] = normalized_port

        # Dynamically recreate namedtuple with normalized fields
        Traffic = namedtuple('Traffic', traffic_dict.keys())

        return Traffic(**traffic_dict)
