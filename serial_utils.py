# serial_utils.py
"""
Cross-platform serial port discovery for the ESP32 node.

The configured port is always tried first. If it is not present, every port
reported by the OS is tried, with likely USB-serial adapters (CP210x, CH340,
FTDI, Silabs) prioritised. This means the same code works on macOS
(``/dev/cu.usbserial-*``), Linux (``/dev/ttyUSB*``, ``/dev/ttyACM*``) and
Windows (``COM*``) without editing config.yaml by hand.
"""

import serial
from serial.tools import list_ports

# Substrings that identify a USB-serial bridge in a port description/hwid.
_USB_HINTS = ('usb', 'uart', 'serial', 'cp210', 'ch340', 'ch341', 'ftdi',
              'silabs', 'wch', 'prolific', 'acm')

# Last-resort names for platforms where enumeration returns nothing useful.
_FALLBACKS = ('/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1')


def _rank(port_info):
    """0 for likely USB serial devices, 1 otherwise."""
    text = f"{port_info.device} {port_info.description} {port_info.hwid}".lower()
    return 0 if any(hint in text for hint in _USB_HINTS) else 1


def candidate_ports(configured_port=None):
    """Return an ordered list of ports to try."""
    candidates = []

    if configured_port:
        candidates.append(configured_port)

    try:
        detected = sorted(list(list_ports.comports()), key=_rank)
    except Exception:
        detected = []

    for port_info in detected:
        if port_info.device not in candidates:
            candidates.append(port_info.device)

    for fallback in _FALLBACKS:
        if fallback not in candidates:
            candidates.append(fallback)

    return candidates


def open_serial(configured_port=None, baud=115200, timeout=1):
    """
    Open the first working serial port.

    Returns
    -------
    (serial.Serial or None, port_name or None)
    """
    for port in candidate_ports(configured_port):
        try:
            return serial.Serial(port, baud, timeout=timeout), port
        except Exception:
            continue
    return None, None


def describe_ports():
    """Human-readable list of detected ports, for diagnostics."""
    try:
        return [f"{p.device} — {p.description}" for p in list_ports.comports()]
    except Exception:
        return []
