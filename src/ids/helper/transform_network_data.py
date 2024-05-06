import datetime
from network_capture import get_ip_address
from typing import List, Dict, Any
from interface import time_str


__all__ = ['transform_network_data']


def transform_network_data(data: List[Dict[str, Any]], INTERFACE: str) -> List[Dict[str, Any]]:
    """
    Transforms raw network traffic data into a structured format by aggregating 
    connection details for each unique IP address encountered. This function identifies
    the host IP dynamically and categorizes other IPs as external or internal based on 
    their relationship with the host IP. It aggregates source and destination ports as 
    well as suspected attack types for each connection.

    Parameters:
        data (List[Dict[str, Any]]): A list of dictionaries where each dictionary represents
            a network traffic record with keys for source and destination addresses and ports,
            and the type of suspected attack if any.

        INTERFACE (str): The network interface to use for dynamically determining the host's 
            IP address. This could be, for example, 'eth0' on Linux or 'en0' on macOS.

    Returns:
        List[Dict[str, Any]]: A transformed list of dictionaries. Each dictionary represents 
            a unique external connection IP with aggregated sets of source ports, destination 
            ports, and suspected attacks associated with that IP.

    Note:
        This function depends on `get_ip_address` to retrieve the host's IP address using the
        specified network interface. Ensure that the INTERFACE parameter is correctly specified
        to reflect the active network interface on the host machine.
    """

    # Dynamically determine host IP
    host_ip = get_ip_address(INTERFACE)
    print(f'{time_str()}: Dynamically determined Host IP: {host_ip}')
    if not host_ip:
        return []

    transformed_data = {}

    for entry in data:
        if entry['src_address'] == host_ip:
            connection_ip = entry['dest_address']
        else:
            connection_ip = entry['src_address']

        if connection_ip not in transformed_data:
            transformed_data[connection_ip] = {'connection_ip': connection_ip, 'from_ports': set(
            ), 'to_ports': set(), 'suspected_attacks': set()}

        if entry['src_address'] == host_ip:
            transformed_data[connection_ip]['from_ports'].add(entry['src_port'])
            transformed_data[connection_ip]['to_ports'].add(entry['dest_port'])
        else:
            transformed_data[connection_ip]['to_ports'].add(entry['src_port'])
            transformed_data[connection_ip]['from_ports'].add(entry['dest_port'])

        transformed_data[connection_ip]['suspected_attacks'].add(
            entry['suspected_attack_type'])

    # Convert aggregate data back into a list of dictionaries
    return list(transformed_data.values())
