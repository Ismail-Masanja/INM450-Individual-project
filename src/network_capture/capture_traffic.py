from scapy.all import sniff
from functools import partial
from .get_ip_address import get_ip_address
from .process_packet import process_packet
from .packets_to_netflow import packets_to_netflow

from interface import time_str


__all__ = ['capture_network_traffic']


def capture_network_traffic(INTERFACE, CAPTURE_DURATION):
    packets_data = []
    seq_nums = {}

    # Try to get the IP address of specified interface
    try:
        HOST_IP = get_ip_address(INTERFACE)
        print(f'{time_str()}: Capturing on interface {INTERFACE} with IP address {HOST_IP}')
    except ValueError as e:
        print(e)
        exit(1)

    _process_packet = partial(process_packet, packets_data, seq_nums, HOST_IP)

    # Start packet capture
    sniff(iface=INTERFACE, prn=_process_packet, store=False, timeout=CAPTURE_DURATION)

    flow_data = packets_to_netflow(packets_data, HOST_IP)

    return flow_data
