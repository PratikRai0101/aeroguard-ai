"""Tests for cross-platform serial port discovery."""

import unittest

import bootstrap_tf  # noqa: F401

from serial_utils import candidate_ports, describe_ports


class SerialUtilsTests(unittest.TestCase):
    def test_configured_port_is_tried_first(self):
        ports = candidate_ports('/dev/cu.usbserial-TEST')
        self.assertEqual(ports[0], '/dev/cu.usbserial-TEST')

    def test_common_fallbacks_are_present(self):
        ports = candidate_ports(None)
        for fallback in ('/dev/ttyUSB0', '/dev/ttyACM0', '/dev/ttyUSB1'):
            self.assertIn(fallback, ports)

    def test_candidates_have_no_duplicates(self):
        ports = candidate_ports('/dev/ttyUSB0')
        self.assertEqual(len(ports), len(set(ports)))

    def test_describe_ports_returns_a_list(self):
        self.assertIsInstance(describe_ports(), list)


if __name__ == '__main__':
    unittest.main()
