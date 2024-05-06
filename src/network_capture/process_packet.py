from datetime import datetime
from scapy.all import IP, TCP, UDP, DNS, DNSQR, DNSRR, ICMP
from typing import List, Dict
from .parse_ftp_response import parse_ftp_response


__all__ = ['process_packet']


def process_packet(packets_data: List[Dict], seq_nums: Dict, HOST_IP: str, packet) -> None:
    """
    Processes a packet to extract various network metrics and information.

    Parameters:
        packets_data (List[Dict]): A list to append the extracted packet information to.
        seq_nums (Dict): A dictionary to track sequence numbers for retransmission detection.
        packet: The packet to be processed.
        HOST_IP (str): The IP address of the host for identifying incoming and outgoing packets.

    Returns:
        None: The function modifies the packets_data list in place, appending a new dictionary
              for each processed packet.
    """

    now = datetime.now()
    if packet.haslayer(IP):
        ip_layer = packet[IP]
        l7_proto = 0
        sport, dport = 0, 0
        tcp_flags = 0
        duration_in = 0
        duration_out = 0
        min_ttl = ip_layer.ttl
        max_ttl = ip_layer.ttl
        longest_flow_pkt = len(packet)
        shortest_flow_pkt = len(packet)
        min_ip_pkt_len = len(packet)
        max_ip_pkt_len = len(packet)
        src_to_dst_second_bytes = 0
        dst_to_src_second_bytes = 0
        retransmitted_in_bytes = 0
        retransmitted_in_pkts = 0
        retransmitted_out_bytes = 0
        retransmitted_out_pkts = 0
        src_to_dst_avg_throughput = 0
        dst_to_src_avg_throughput = 0
        num_pkts_up_to_128_bytes = 0
        num_pkts_128_to_256_bytes = 0
        num_pkts_256_to_512_bytes = 0
        num_pkts_512_to_1024_bytes = 0
        num_pkts_1024_to_1514_bytes = 0
        tcp_win_size = 0
        tcp_win_max_in = 0
        tcp_win_max_out = 0
        dns_query_id = 0
        dns_query_type = 0
        dns_ttl_answer = 0
        ftp_command_ret_code = 0
        retransmitted_in_bytes, retransmitted_in_pkts = 0, 0
        retransmitted_out_bytes, retransmitted_out_pkts = 0, 0

        if packet.haslayer(TCP):
            transport_layer = packet[TCP]
            sport, dport = transport_layer.sport, transport_layer.dport
            tcp_flags = int(transport_layer.flags)
            l7_proto = transport_layer.dport
            tcp_win_size = transport_layer.window
            flow_key = (ip_layer.src, transport_layer.sport, ip_layer.dst,
                        transport_layer.dport, ip_layer.proto)
            is_outgoing = ip_layer.src == HOST_IP

            # Check for retransmission by comparing sequence numbers
            if flow_key in seq_nums:
                if transport_layer.seq <= seq_nums[flow_key]:
                    if is_outgoing:
                        retransmitted_out_bytes, retransmitted_out_pkts = len(packet), 1
                    else:
                        retransmitted_in_bytes, retransmitted_in_pkts = len(packet), 1
            else:
                seq_nums[flow_key] = transport_layer.seq

        elif packet.haslayer(UDP):
            transport_layer = packet[UDP]
            sport, dport = transport_layer.sport, transport_layer.dport
            l7_proto = transport_layer.dport

        if packet.haslayer(DNS):
            dns_layer = packet[DNS]
            dns_query_id = dns_layer.id
            if dns_layer.qr == 0:  # This is a query
                dns_query_type = dns_layer[DNSQR].qtype
            elif dns_layer.qr == 1:  # This is a response
                # Assuming only one answer, get TTL first answer
                if dns_layer.ancount > 0:
                    dns_ttl_answer = dns_layer[DNSRR][0].ttl

        if packet.haslayer(TCP) and (packet[TCP].dport == 21 or packet[TCP].sport == 21):
            # Extract payload from TCP segment
            payload = packet[TCP].payload
            ftp_command_ret_code = parse_ftp_response(payload)

        # Determine if packet is incoming or outgoing
        is_outgoing = ip_layer.src == HOST_IP

        # Determine direction of TCP flags based on packet direction
        if is_outgoing:
            client_tcp_flags = tcp_flags
            server_tcp_flags = 0
        else:
            client_tcp_flags = 0
            server_tcp_flags = tcp_flags

        if packet.haslayer(ICMP):
            icmp_layer = packet[ICMP]
            icmp_type = icmp_layer.type
            icmp_code = icmp_layer.code
        else:
            icmp_type = 0
            icmp_code = 0

        packet_size = len(packet)
        if packet_size <= 128:
            num_pkts_up_to_128_bytes += 1
        elif 128 < packet_size <= 256:
            num_pkts_128_to_256_bytes += 1
        elif 256 < packet_size <= 512:
            num_pkts_256_to_512_bytes += 1
        elif 512 < packet_size <= 1024:
            num_pkts_512_to_1024_bytes += 1
        elif 1024 < packet_size <= 1514:
            num_pkts_1024_to_1514_bytes += 1

        packet_info = {
            "IPV4_SRC_ADDR": ip_layer.src,
            "L4_SRC_PORT": sport,
            "IPV4_DST_ADDR": ip_layer.dst,
            "L4_DST_PORT": dport,
            "PROTOCOL": ip_layer.proto,
            "L7_PROTO": l7_proto,
            "IN_BYTES": 0,
            "IN_PKTS": 0,
            "OUT_BYTES": 0,
            "OUT_PKTS": 0,
            "TCP_FLAGS": tcp_flags,
            "CLIENT_TCP_FLAGS": client_tcp_flags,
            "SERVER_TCP_FLAGS": server_tcp_flags,
            "DURATION_IN": duration_in,
            "DURATION_OUT": duration_out,
            "MIN_TTL": min_ttl,
            "MAX_TTL": max_ttl,
            "LONGEST_FLOW_PKT": longest_flow_pkt,
            "SHORTEST_FLOW_PKT": shortest_flow_pkt,
            "MIN_IP_PKT_LEN": min_ip_pkt_len,
            "MAX_IP_PKT_LEN": max_ip_pkt_len,
            "SRC_TO_DST_SECOND_BYTES": src_to_dst_second_bytes,
            "DST_TO_SRC_SECOND_BYTES": dst_to_src_second_bytes,
            "RETRANSMITTED_IN_BYTES": retransmitted_in_bytes,
            "RETRANSMITTED_IN_PKTS": retransmitted_in_pkts,
            "RETRANSMITTED_OUT_BYTES": retransmitted_out_bytes,
            "RETRANSMITTED_OUT_PKTS": retransmitted_out_pkts,
            "SRC_TO_DST_AVG_THROUGHPUT": src_to_dst_avg_throughput,
            "DST_TO_SRC_AVG_THROUGHPUT": dst_to_src_avg_throughput,
            "NUM_PKTS_UP_TO_128_BYTES": num_pkts_up_to_128_bytes,
            "NUM_PKTS_128_TO_256_BYTES": num_pkts_128_to_256_bytes,
            "NUM_PKTS_256_TO_512_BYTES": num_pkts_256_to_512_bytes,
            "NUM_PKTS_512_TO_1024_BYTES": num_pkts_512_to_1024_bytes,
            "NUM_PKTS_1024_TO_1514_BYTES": num_pkts_1024_to_1514_bytes,
            "TCP_WIN_MAX_IN": tcp_win_max_in,
            "TCP_WIN_MAX_OUT": tcp_win_max_out,
            "ICMP_TYPE": icmp_type,
            "ICMP_IPV4_TYPE": icmp_code,
            "DNS_QUERY_ID": dns_query_id,
            "DNS_QUERY_TYPE": dns_query_type,
            "DNS_TTL_ANSWER": dns_ttl_answer,
            "FTP_COMMAND_RET_CODE": ftp_command_ret_code,
            "PACKET_LENGTH": len(packet),
            "TIMESTAMP": now,
        }
        if is_outgoing:
            packet_info["OUT_BYTES"] = len(packet)
            packet_info["OUT_PKTS"] = 1
            packet_info["TCP_WIN_MAX_OUT"] = max(
                packet_info.get("TCP_WIN_MAX_OUT", 0), tcp_win_size)
        else:
            packet_info["IN_BYTES"] = len(packet)
            packet_info["IN_PKTS"] = 1
            packet_info["TCP_WIN_MAX_IN"] = max(
                packet_info.get("TCP_WIN_MAX_IN", 0), tcp_win_size)

        packets_data.append(packet_info)
