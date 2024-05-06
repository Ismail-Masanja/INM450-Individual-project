import subprocess
import platform
import tempfile
import os
from interface import time_str
from typing import Literal


__all__ = ['block_ip']


def block_ip(ip_address: str, action: Literal["block", "unblock"]) -> None:
    """
    Blocks or unblocks an IP address on the host machine.

    Args:
        ip_address (str): The IP address to block or unblock.
        action (Literal["block", "unblock"]): The action to perform. Either "block" or "unblock".

    Raises:
        ValueError: If an unsupported action is passed.
        RuntimeError: If the operating system is not supported.
    """

    os_name = platform.system()

    if os_name == "Linux":
        block_ip_linux(ip_address, action)
    elif os_name == "Windows":
        block_ip_windows(ip_address, action)
    elif os_name == "Darwin":  # macOS
        block_ip_macos(ip_address, action)
    else:
        print(f'{time_str()}: OS {os_name} is not supported for this operation.')


def block_ip_linux(ip_address: str, action: Literal["block", "unblock"]) -> None:
    """
    Blocks or unblocks an IP address on a Linux system.

    Args:
        ip_address (str): The IP address to block or unblock.
        action (Literal["block", "unblock"]): The action to perform.
    """

    if action == "block":
        command = ["sudo", "iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"]
    else:  # action == "unblock"
        command = ["sudo", "iptables", "-D", "INPUT", "-s", ip_address, "-j", "DROP"]
    subprocess.run(command)
    print(f'{time_str()}: {action.capitalize()}ed IP address {ip_address} on Linux.')


def block_ip_windows(ip_address: str, action: Literal["block", "unblock"]) -> None:
    """
    Blocks or unblocks an IP address on a Windows system.

    Args:
        ip_address (str): The IP address to block or unblock.
        action (Literal["block", "unblock"]): The action to perform.
    """

    if action == "block":
        command = f"netsh advfirewall firewall add rule name=\"BlockIP_{ip_address}\" dir=in action=block remoteip={ip_address}"
    else:  # action == "unblock"
        command = f"netsh advfirewall firewall delete rule name=\"BlockIP_{ip_address}\""
    subprocess.run(command, shell=True)
    print(f'{time_str()}: {action.capitalize()}ed IP address {ip_address} on Windows.')


def block_ip_macos(ip_address: str, action: Literal["block", "unblock"]) -> None:
    """
    Blocks or unblocks an IP address on a macOS system.

    Args:
        ip_address (str): The IP address to block or unblock.
        action (Literal["block", "unblock"]): The action to perform.
    """

    original_pf_conf_path = "/etc/pf.conf"

    # Create a temporary pf.conf with modifications
    with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as temp_pf_conf:
        # Rule to block the IP address
        ip_rule = f"block drop from any to {ip_address}\n"

        # Read the original pf.conf and decide whether to add or remove the block rule
        with open(original_pf_conf_path, "r") as original_pf_conf:
            lines = original_pf_conf.readlines()
            if action == "block" and ip_rule not in lines:
                lines.append(ip_rule)
            elif action == "unblock" and ip_rule in lines:
                lines.remove(ip_rule)

        # Write the modified configuration to the temporary file
        temp_pf_conf.writelines(lines)
        temp_pf_conf_path = temp_pf_conf.name

    # Load the modified configuration using pfctl
    try:
        # subprocess.run(["sudo", "pfctl", "-e"], check=True)
        subprocess.run(["sudo", "pfctl", "-f", temp_pf_conf_path], check=True)
        print(f'{time_str()}: {action.capitalize()}ed IP address {ip_address} on macOS.')
    except subprocess.CalledProcessError as e:
        print(f'{time_str()}: Failed to {action} IP address {ip_address} on macOS: {e}')
    finally:
        # Clean up by deleting the temporary file
        os.remove(temp_pf_conf_path)
