import json
import socket

import pytest

from target_finder_toolkit.bubblegazelens import UdpGazeProvider


class FakeClock:
    def __init__(self, seconds: float = 100.0) -> None:
        self.seconds = seconds

    def __call__(self) -> float:
        return self.seconds


def _send(port: int, payload) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        sender.sendto(data, ("127.0.0.1", port))


def test_udp_provider_accepts_primary_screen_logical_pixels():
    clock = FakeClock()
    provider = UdpGazeProvider(0, clock=clock)
    try:
        _send(
            provider.port,
            {"t_ms": 1234.5, "x": 812.2, "y": 498.1, "valid": True, "screen": "primary"},
        )

        sample = provider.get_sample()

        assert sample.t_ms == 0.0
        assert sample.x == 812.2
        assert sample.y == 498.1
        assert sample.valid
        assert provider.diagnostics()["source_t_ms"] == 1234.5
        assert provider.diagnostics()["bind"].startswith("127.0.0.1:")
    finally:
        provider.close()


def test_udp_provider_marks_a_silent_stream_invalid():
    clock = FakeClock()
    provider = UdpGazeProvider(0, hold_valid_ms=50, clock=clock)
    try:
        _send(
            provider.port,
            {"t_ms": 0, "x": 300, "y": 400, "valid": True, "screen": "primary"},
        )
        assert provider.get_sample().valid

        clock.seconds += 0.051
        sample = provider.get_sample()

        assert not sample.valid
        assert sample.t_ms == pytest.approx(51.0)
        assert sample.x == 300
        assert sample.y == 400
    finally:
        provider.close()


def test_udp_provider_rejects_ambiguous_or_malformed_coordinate_contracts():
    clock = FakeClock()
    provider = UdpGazeProvider(0, clock=clock)
    try:
        _send(
            provider.port,
            {"t_ms": 1, "x": 0.5, "y": 0.25, "valid": True, "screen": "primary"},
        )
        _send(
            provider.port,
            {"t_ms": 2, "x": 500, "y": 250, "valid": True, "screen": "secondary"},
        )
        _send(provider.port, b"not-json")

        sample = provider.get_sample()
        diagnostics = provider.diagnostics()

        assert not sample.valid
        assert diagnostics["received_packets"] == 0
        assert diagnostics["dropped_packets"] == 3
        assert diagnostics["last_drop_reason"] == "invalid_json"
    finally:
        provider.close()


def test_udp_provider_rejects_backward_source_timestamps():
    clock = FakeClock()
    provider = UdpGazeProvider(0, clock=clock)
    try:
        _send(
            provider.port,
            {"t_ms": 10, "x": 500, "y": 250, "valid": True, "screen": "primary"},
        )
        assert provider.get_sample().x == 500
        _send(
            provider.port,
            {"t_ms": 9, "x": 900, "y": 250, "valid": True, "screen": "primary"},
        )

        sample = provider.get_sample()
        diagnostics = provider.diagnostics()

        assert sample.x == 500
        assert diagnostics["received_packets"] == 1
        assert diagnostics["dropped_packets"] == 1
        assert diagnostics["last_drop_reason"] == "source_timestamp_moved_backwards"
    finally:
        provider.close()


def test_udp_provider_accepts_explicit_invalidity_without_coordinates():
    clock = FakeClock()
    provider = UdpGazeProvider(0, clock=clock)
    try:
        _send(
            provider.port,
            {"t_ms": 12, "valid": False, "screen": "primary"},
        )

        sample = provider.get_sample()

        assert not sample.valid
        assert provider.diagnostics()["received_packets"] == 1
    finally:
        provider.close()
