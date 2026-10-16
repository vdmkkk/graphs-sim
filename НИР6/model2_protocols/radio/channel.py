"""Shared half-duplex radio medium with collisions and capture effect.

LoRa is a *broadcast* medium: a transmission is heard by every node in range
simultaneously, and two overlapping transmissions on the same channel
interfere. This module models that medium at the frame level, which is what
makes the MAC realistic enough to expose the duty-cycle / overhead / collision
trade-offs between routing protocols.

Collision model (standard for LoRa simulators such as LoRaSim / FLoRa):
  * Two frames overlapping in time at a common receiver collide.
  * Capture effect: the stronger frame still decodes if it exceeds every
    overlapping interferer by ``capture_threshold_db`` at that receiver;
    otherwise both are lost.
  * A surviving frame is then subject to a probabilistic SNR draw (fading /
    residual symbol errors near the sensitivity edge).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import MACConfig
from .lora_phy import reception_probability


@dataclass
class Transmission:
    """A frame in flight on the channel."""

    tx_id: int
    sender: int
    frame: object          # the routing-layer frame being carried
    start: float
    end: float
    payload_bytes: int
    # Receivers at which this frame has been knocked out by a collision.
    lost_at: set[int] = field(default_factory=set)
    # Cached sampled (rssi, snr) per receiver for this specific transmission.
    link_cache: dict[int, tuple[float, float]] = field(default_factory=dict)


class Channel:
    """Tracks in-flight transmissions and decides per-receiver reception.

    The ``network`` collaborator must provide::

        feasible_receivers(sender_id) -> list[int]
        sample_link(sender_id, receiver_id, rng) -> (rssi_dbm, snr_db)

    so the channel stays agnostic of geometry and node bookkeeping.
    """

    def __init__(self, network, mac: MACConfig, phy):
        self.network = network
        self.mac = mac
        self.phy = phy
        self.active: dict[int, Transmission] = {}
        self._next_id = 0
        # Diagnostics for the metrics layer.
        self.total_transmissions = 0
        self.total_collisions = 0       # per-receiver knock-outs (can exceed tx)
        self.collided_transmissions = 0  # transmissions losing >=1 receiver

    # ------------------------------------------------------------------
    def is_busy(self, listener: int, rng) -> bool:
        """Listen-before-talk: does *listener* currently hear any carrier?"""
        for tx in self.active.values():
            if tx.sender == listener:
                continue
            rssi, _ = self._link(tx, listener, rng)
            if rssi >= self.phy.sensitivity():
                return True
        return False

    # ------------------------------------------------------------------
    def begin(self, sender: int, frame: object, start: float, end: float,
              payload_bytes: int, rng) -> Transmission:
        """Register a new transmission and resolve collisions vs. active ones."""
        tx = Transmission(self._next_id, sender, frame, start, end, payload_bytes)
        self._next_id += 1
        self.total_transmissions += 1

        receivers = set(self.network.feasible_receivers(sender))

        # Resolve interference against every currently-active transmission.
        # Any active transmission necessarily overlaps this one in time
        # (it started earlier and has not finished), so the intersection of
        # their receiver sets is exactly where they contend.
        for other in self.active.values():
            other_receivers = set(self.network.feasible_receivers(other.sender))
            for r in receivers & other_receivers:
                rssi_new, _ = self._link(tx, r, rng)
                rssi_old, _ = self._link(other, r, rng)
                self._apply_capture(tx, other, r, rssi_new, rssi_old)

        self.active[tx.tx_id] = tx
        return tx

    def _apply_capture(self, a: Transmission, b: Transmission, r: int,
                       rssi_a: float, rssi_b: float) -> None:
        """Decide which of two overlapping frames (if any) survives at r."""
        thr = self.mac.capture_threshold_db
        if rssi_a >= rssi_b + thr:
            # a captures the channel; b is lost at r.
            if r not in b.lost_at:
                b.lost_at.add(r)
                self.total_collisions += 1
        elif rssi_b >= rssi_a + thr:
            if r not in a.lost_at:
                a.lost_at.add(r)
                self.total_collisions += 1
        else:
            # Neither dominates -> both destroyed at r.
            for tx, was in ((a, r in a.lost_at), (b, r in b.lost_at)):
                if not was:
                    tx.lost_at.add(r)
                    self.total_collisions += 1

    # ------------------------------------------------------------------
    def finish(self, tx: Transmission, rng) -> list[tuple[int, float, float]]:
        """Conclude a transmission; return [(receiver, rssi, snr), ...] decoded.

        Half-duplex (a receiver that was itself transmitting) is filtered by
        the engine, which owns node radio state; here we apply collisions and
        the probabilistic SNR draw.
        """
        self.active.pop(tx.tx_id, None)
        if tx.lost_at:
            self.collided_transmissions += 1
        decoded: list[tuple[int, float, float]] = []
        for r in self.network.feasible_receivers(tx.sender):
            if r in tx.lost_at:
                continue
            rssi, snr = self._link(tx, r, rng)
            p = reception_probability(snr, rssi, self.phy)
            if p <= 0.0:
                continue
            if p >= 1.0 or rng.random() < p:
                decoded.append((r, rssi, snr))
        return decoded

    # ------------------------------------------------------------------
    def _link(self, tx: Transmission, receiver: int, rng) -> tuple[float, float]:
        cached = tx.link_cache.get(receiver)
        if cached is None:
            cached = self.network.sample_link(tx.sender, receiver, rng)
            tx.link_cache[receiver] = cached
        return cached
