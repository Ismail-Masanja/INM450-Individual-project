import netifaces as ni
from typing import Optional

__all__ = ['get_ip_address']


def get_ip_address(interface: str) -> Optional[str]:
    """
    Retrieves the IPv4 address associated with a given network interface.

    Parameters:
        interface (str): The name of the network interface whose IP address is to be retrieved.

    Returns:
        Optional[str]: The IPv4 address of the specified network interface, or None if the interface does not have an IPv4 address.

    Raises:
        ValueError: If no IPv4 address is found for the specified network interface.
        ValueError: If the specified network interface does not exist.
    """
    try:
        # Retrieve addresses for specified network interface
        addrs = ni.ifaddresses(interface)
    except ValueError as e:
        # The specified interface does not exist
        raise ValueError(f"The network interface '{interface}' does not exist.") from e

    # Attempt to get IPv4 address
    ip_info = addrs.get(ni.AF_INET)

    if ip_info:
        # Return first IPv4 address found
        return ip_info[0]['addr']
    else:
        # No IPv4 address found for interface
        raise ValueError(f"No IPv4 address found for interface '{interface}'.")
