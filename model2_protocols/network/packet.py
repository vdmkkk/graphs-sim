"""Frames (link-layer PDUs) and application Messages.

A *Message* is what the application wants delivered end-to-end. A *Frame*
is one over-the-air transmission; depending on the protocol a message may
be carried by many frames (re-broadcasts, route requests, hop-by-hop relays).
Separating them lets the metrics layer distinguish *goodput* (delivered
messages) from *overhead* (control/duplicate frames on the air).
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field


class FrameType(enum.Enum):
    DATA = "DATA"          # carries application payload
    RREQ = "RREQ"          # AODV route request
    RREP = "RREP"          # AODV route reply
    RERR = "RERR"          # AODV route error
    HELLO = "HELLO"        # neighbour discovery (OLSR / proactive)
    TC = "TC"              # OLSR topology control
    ACK = "ACK"            # link acknowledgement


@dataclass
class Message:
    """An end-to-end application message (the unit PDR is measured on)."""

    msg_id: int
    src: int
    dst: int
    created_at: float
    size_bytes: int
    delivered_at: float | None = None
    delivered: bool = False
    hops: int = 0

    @property
    def latency(self) -> float | None:
        if self.delivered_at is None:
            return None
        return self.delivered_at - self.created_at


_frame_counter = itertools.count()


@dataclass
class Frame:
    """One over-the-air PDU. ``next_hop`` = -1 means broadcast to all."""

    ftype: FrameType
    src: int                 # originator of the conversation
    dst: int                 # ultimate destination (-1 for network-wide)
    sender: int              # node currently (re)transmitting this frame
    next_hop: int            # intended receiver (-1 = any / broadcast)
    payload_bytes: int
    msg_id: int = -1         # links DATA frames back to their Message
    hop_count: int = 0       # hops travelled so far
    ttl: int = 8             # remaining hop budget
    is_control: bool = False  # True for routing-overhead frames
    seq: int = -1            # protocol sequence number (dedup / freshness)
    frame_id: int = field(default_factory=lambda: next(_frame_counter))
    # Free-form protocol scratch space (e.g. AODV hop metric, route record).
    meta: dict = field(default_factory=dict)

    def copy_for_relay(self, sender: int, next_hop: int) -> "Frame":
        """Clone this frame for re-transmission by *sender* (new frame id)."""
        return Frame(
            ftype=self.ftype,
            src=self.src,
            dst=self.dst,
            sender=sender,
            next_hop=next_hop,
            payload_bytes=self.payload_bytes,
            msg_id=self.msg_id,
            hop_count=self.hop_count + 1,
            ttl=self.ttl - 1,
            is_control=self.is_control,
            seq=self.seq,
            meta=dict(self.meta),
        )
