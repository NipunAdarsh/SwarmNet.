"""
SwarmNet — UDP Multicast Node Discovery

Provides automatic plug-and-play node registration.  On startup each node
broadcasts a presence packet over UDP multicast.  The registry listener
picks up these packets and auto-registers the node.

For single-machine demos, a localhost fallback is included.
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger("swarm.discovery")

# Multicast group and port for swarm discovery
MULTICAST_GROUP = "239.1.2.3"
MULTICAST_PORT = 5007

# How often to re-broadcast presence (seconds)
BROADCAST_INTERVAL_S = 5.0


class DiscoveryBroadcaster:
    """Broadcasts UDP multicast presence packets for node auto-discovery.

    Packet payload (JSON):
    {
        "device_id": "node-abc123",
        "ip_address": "192.168.1.42",
        "port": 8000,
        "cpu_cores": 8,
        "npu_available": true,
        "gpu_available": false,
        "memory_mb": 16384,
        "available_accelerators": ["QNNExecutionProvider"]
    }
    """

    def __init__(self, node_info: dict, multicast_group: str = MULTICAST_GROUP,
                 port: int = MULTICAST_PORT) -> None:
        self._info = node_info
        self._group = multicast_group
        self._port = port
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._broadcast_loop, daemon=True,
            name=f"discovery-broadcast-{self._info.get('device_id', '?')}",
        )
        self._thread.start()
        logger.info(
            "Discovery broadcaster started for %s → %s:%d",
            self._info.get("device_id"), self._group, self._port,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _broadcast_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        # Allow localhost multicast
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

        payload = json.dumps(self._info).encode("utf-8")

        while self._running:
            try:
                sock.sendto(payload, (self._group, self._port))
            except Exception as exc:
                logger.debug("Broadcast send error: %s", exc)
            time.sleep(BROADCAST_INTERVAL_S)

        sock.close()


class DiscoveryListener:
    """Listens for UDP multicast presence packets and auto-registers nodes.

    Parameters
    ----------
    on_node_discovered : callable
        Function called with the parsed node info dict when a new node
        broadcasts its presence.
    """

    def __init__(self, on_node_discovered: Callable[[dict], None],
                 multicast_group: str = MULTICAST_GROUP,
                 port: int = MULTICAST_PORT) -> None:
        self._callback = on_node_discovered
        self._group = multicast_group
        self._port = port
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="discovery-listener",
        )
        self._thread.start()
        logger.info("Discovery listener started on %s:%d", self._group, self._port)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _listen_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            sock.bind(("", self._port))
        except Exception as exc:
            logger.warning("Could not bind discovery listener: %s", exc)
            return

        # Join multicast group
        mreq = struct.pack(
            "4sL", socket.inet_aton(self._group), socket.INADDR_ANY
        )
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except Exception as exc:
            logger.warning("Could not join multicast group: %s", exc)

        sock.settimeout(2.0)

        while self._running:
            try:
                data, addr = sock.recvfrom(4096)
                info = json.loads(data.decode("utf-8"))
                logger.debug("Discovery packet from %s: %s", addr, info.get("device_id"))
                self._callback(info)
            except socket.timeout:
                continue
            except Exception as exc:
                logger.debug("Discovery listener error: %s", exc)
                time.sleep(0.5)

        sock.close()
        logger.info("Discovery listener stopped")
