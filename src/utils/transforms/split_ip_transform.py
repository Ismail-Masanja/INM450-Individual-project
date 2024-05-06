from collections import namedtuple


__all__ = ['SplitIPTransform']


class SplitIPTransform:
    """Transform that splits IP addresses into octets and normalizes them to the range [-1, 1].

    This transform takes a namedtuple representing network traffic data, splits the IPv4
    source and destination addresses into their constituent octets, normalizes these octets
    to a range of [-1, 1], and then adds these as new fields to the data while removing the
    original IP address fields.
    """

    def __call__(self, traffic: namedtuple) -> namedtuple:
        """Apply the IP split and normalization transform to a given traffic data sample.

        Args:
            traffic (namedtuple): A namedtuple representing a single sample of network
                                  traffic data, containing IPv4 source and destination
                                  address fields among potentially others.

        Returns:
            namedtuple: A new namedtuple instance with the original IP address fields
                        replaced by their split and normalized octet fields.
        """
        # Convert namedtuple to dictionary for manipulation
        traffic_dict = traffic._asdict()

        for ip_field in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']:
            if ip_field in traffic_dict:
                # Split IP address into octets and normalize each
                ip_octets = traffic_dict[ip_field].split('.')
                # Normalize and add new fields for each octet
                for i, octet in enumerate(ip_octets):
                    normalized_octet = (float(octet) / 255.0) * 2 - 1
                    traffic_dict[f"{ip_field}_OCTET_{i+1}"] = normalized_octet
                # Remove the original IP address field
                del traffic_dict[ip_field]

        # Dynamically recreate namedtuple with new fields
        Traffic = namedtuple('Traffic', traffic_dict.keys())

        return Traffic(**traffic_dict)
