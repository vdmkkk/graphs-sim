"""Routing-protocol interface and shared helpers.

A protocol is driven entirely by three engine callbacks:

    on_start(sim)                         once, before traffic begins
    on_app_message(sim, node, message)    this node originates a message
    on_receive(sim, node, frame, rssi, snr)   a frame was decoded at node

Per-node protocol state lives in ``node.proto`` so a single protocol object
can serve the whole network without leaking state between nodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..network.packet import Frame, FrameType, Message


class RoutingProtocol(ABC):
    #: human-readable name used in result tables
    name: str = "base"
    #: proactive protocols keep routes fresh continuously (control overhead)
    proactive: bool = False

    def __init__(self, cfg):
        self.cfg = cfg

    # -- engine callbacks ---------------------------------------------------
    def on_start(self, sim) -> None:  # noqa: D401 - optional hook
        """Schedule any periodic control traffic. Default: none."""

    @abstractmethod
    def on_app_message(self, sim, node, message: Message) -> None: ...

    @abstractmethod
    def on_receive(self, sim, node, frame: Frame, rssi: float, snr: float) -> None: ...

    # -- shared helpers -----------------------------------------------------
    def _seen_key(self, node) -> set:
        return node.proto.setdefault("seen", set())

    def already_seen(self, node, key) -> bool:
        """Idempotent dedup: returns True if *key* was seen before this call."""
        seen = self._seen_key(node)
        if key in seen:
            return True
        seen.add(key)
        return False

    def deliver_if_dst(self, sim, node, frame: Frame) -> bool:
        """If this node is the destination, mark the message delivered."""
        if frame.dst == node.id and frame.ftype is FrameType.DATA:
            # hop_count counts relays; +1 accounts for the originator's own
            # first transmission, so a direct-neighbour delivery reads as 1 hop.
            sim.metrics.on_delivered(frame.msg_id, sim.now, frame.hop_count + 1)
            return True
        return False

    def make_data_frame(self, message: Message, ttl: int, next_hop: int = -1) -> Frame:
        return Frame(
            ftype=FrameType.DATA,
            src=message.src,
            dst=message.dst,
            sender=message.src,
            next_hop=next_hop,
            payload_bytes=self.cfg.data_payload_bytes + self.cfg.header_bytes,
            msg_id=message.msg_id,
            ttl=ttl,
            is_control=False,
        )
