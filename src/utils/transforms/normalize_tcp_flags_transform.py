from collections import namedtuple


__all__ = ['NormalizeTCPFlagsTransform']


class NormalizeTCPFlagsTransform:
    """Transform class for normalizing TCP flags to the range [-1, 1].

    This class takes a network traffic data sample represented as a namedtuple and normalizes
    the values of TCP flags (TCP_FLAGS, CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS) based on the maximum
    value a TCP flag can have (255 for 8 bits) to a range of [-1, 1].

    After normalization, it returns a new namedtuple with the normalized TCP flag values.
    """

    def __call__(self, traffic: namedtuple) -> namedtuple:
        """Normalize the TCP flag fields of a traffic data sample.

        Args:
            traffic (namedtuple): A namedtuple representing a single sample of network
                                  traffic data, containing TCP flag fields among potentially
                                  others.

        Returns:
            namedtuple: A new namedtuple instance with the TCP flag fields normalized to
                        the range [-1, 1].
        """
        # Convert namedtuple to dictionary for manipulation
        traffic_dict = traffic._asdict()

        # Fields to normalize
        tcp_flag_fields = ['TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS']

        for field in tcp_flag_fields:
            if field in traffic_dict:
                # Normalize the field
                normalized_value = (float(traffic_dict[field]) / 127.5) - 1
                traffic_dict[field] = normalized_value

        # Dynamically recreate namedtuple with normalized fields
        Traffic = namedtuple('Traffic', traffic_dict.keys())

        return Traffic(**traffic_dict)
