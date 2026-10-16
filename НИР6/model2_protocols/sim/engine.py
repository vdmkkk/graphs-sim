"""Discrete-event simulation engine and the shared MAC layer.

All protocols run on top of *one* MAC so the comparison is fair: they differ
only in routing logic, not in channel access. The MAC implements:

  * listen-before-talk (CSMA) with random exponential-ish backoff,
  * EU868 duty-cycle enforcement (a frame that would breach the 1 % budget
    is dropped — this is what punishes chatty proactive protocols),
  * half-duplex radios (a node cannot hear while it transmits),
  * collisions + capture, delegated to :class:`Channel`.

Time is event-driven: nothing is polled, so a one-hour simulation costs only
as many steps as there are real radio events.
"""

from __future__ import annotations

import heapq
import itertools
import random

from ..config import SimConfig
from ..network.deployment import Network
from ..network.packet import Frame, FrameType
from ..radio.channel import Channel
from ..radio.lora_phy import time_on_air


class Simulator:
    def __init__(self, network: Network, cfg: SimConfig, protocol, metrics,
                 rng: random.Random | None = None):
        self.net = network
        self.cfg = cfg
        self.protocol = protocol
        self.metrics = metrics
        self.rng = rng or random.Random(cfg.seed)
        self.channel = Channel(network, cfg.mac, cfg.phy)

        self.now = 0.0
        self._pq: list[tuple[float, int, object, tuple]] = []
        self._seq = itertools.count()
        self._stopped = False
        self._acks: dict[tuple[int, int], dict] = {}  # (sender, frame_id) -> entry

    # ------------------------------------------------------------------
    # scheduling
    # ------------------------------------------------------------------
    def schedule(self, delay: float, fn, *args) -> None:
        heapq.heappush(self._pq, (self.now + max(delay, 0.0), next(self._seq), fn, args))

    def schedule_at(self, when: float, fn, *args) -> None:
        heapq.heappush(self._pq, (max(when, self.now), next(self._seq), fn, args))

    def schedule_timer(self, node_id: int, delay: float, callback, *cbargs) -> None:
        """Protocol timer; ``callback(sim, node, *cbargs)`` fires after *delay*.

        The callback is invoked regardless of node liveness so that periodic
        chains (HELLO/TC) survive transient failures and re-arm themselves;
        callbacks guard their own radio activity (``mac_send`` already drops
        when the node is down).
        """
        self.schedule(delay, self._fire_timer, node_id, callback, cbargs)

    def _fire_timer(self, node_id: int, callback, cbargs) -> None:
        node = self.net.nodes[node_id]
        callback(self, node, *cbargs)

    # ------------------------------------------------------------------
    # MAC: transmit a frame from a node
    # ------------------------------------------------------------------
    def mac_send(self, node_id: int, frame: Frame, tries: int = 0) -> None:
        node = self.net.nodes[node_id]
        if not node.is_operational(self.now):
            return

        airtime = time_on_air(frame.payload_bytes, self.cfg.phy)

        # --- listen-before-talk -------------------------------------------------
        if self.cfg.mac.listen_before_talk and self.channel.is_busy(node_id, self.rng):
            if tries < self.cfg.mac.max_backoff_tries:
                backoff = self.rng.uniform(self.cfg.mac.backoff_min_s,
                                           self.cfg.mac.backoff_max_s) * (1 + tries)
                self.schedule(backoff, self.mac_send, node_id, frame, tries + 1)
            else:
                self.metrics.on_mac_drop(frame, reason="channel_busy")
            return

        # --- duty-cycle enforcement --------------------------------------------
        if not node.duty_allows(self.now, airtime):
            self.metrics.on_mac_drop(frame, reason="duty_cycle")
            return

        # --- transmit ----------------------------------------------------------
        start, end = self.now, self.now + airtime
        node.record_transmission(start, end, airtime)
        self.metrics.on_air(frame, airtime, self.now)
        tx = self.channel.begin(node_id, frame, start, end, frame.payload_bytes, self.rng)
        self.schedule_at(end, self._on_tx_end, tx)

        # --- arm ARQ for reliable unicast --------------------------------------
        if (self.cfg.mac.use_arq and frame.next_hop != -1
                and frame.ftype is not FrameType.ACK):
            key = (node_id, frame.frame_id)
            entry = self._acks.get(key)
            if entry is None:
                entry = {"frame": frame, "tries": 0, "acked": False}
                self._acks[key] = entry
            self.schedule(self.cfg.mac.ack_timeout_s, self._ack_timeout, key)

    def _on_tx_end(self, tx) -> None:
        frame = tx.frame
        decoded = self.channel.finish(tx, self.rng)
        for receiver, rssi, snr in decoded:
            rnode = self.net.nodes[receiver]
            if not rnode.is_operational(self.now):
                continue
            if rnode.transmitted_during(tx.start, tx.end):
                continue  # half-duplex: was transmitting, deaf to this frame
            rnode.record_reception()
            self.metrics.on_receive(frame, receiver, self.now)
            # The addressed next hop of a reliable unicast frame ACKs it.
            if (self.cfg.mac.use_arq and frame.next_hop == receiver
                    and frame.ftype is not FrameType.ACK):
                self.schedule(self.cfg.mac.sifs_s, self._send_ack, receiver, frame)
            # store-and-forward processing delay before the routing layer acts
            self.schedule(self.cfg.mac.proc_delay_s,
                          self._deliver, receiver, frame, rssi, snr)

    def _send_ack(self, receiver: int, data_frame: Frame) -> None:
        ack = Frame(
            ftype=FrameType.ACK, src=receiver, dst=data_frame.sender,
            sender=receiver, next_hop=data_frame.sender,
            payload_bytes=self.cfg.mac.ack_bytes, is_control=True,
            meta={"ack_id": data_frame.frame_id},
        )
        self.mac_send(receiver, ack)

    def _ack_timeout(self, key) -> None:
        entry = self._acks.get(key)
        if entry is None or entry["acked"]:
            self._acks.pop(key, None)
            return
        if entry["tries"] < self.cfg.mac.max_data_retx:
            entry["tries"] += 1
            self.mac_send(key[0], entry["frame"])  # retransmit (reschedules timeout)
        else:
            self.metrics.on_mac_drop(entry["frame"], reason="no_ack")
            self._acks.pop(key, None)

    def _deliver(self, node_id: int, frame: Frame, rssi: float, snr: float) -> None:
        node = self.net.nodes[node_id]
        if not node.is_operational(self.now):
            return
        if frame.ftype is FrameType.ACK:
            if frame.next_hop == node_id:
                entry = self._acks.get((node_id, frame.meta["ack_id"]))
                if entry is not None:
                    entry["acked"] = True
            return  # ACKs never reach the routing layer
        self.protocol.on_receive(self, node, frame, rssi, snr)

    # ------------------------------------------------------------------
    # application traffic
    # ------------------------------------------------------------------
    def app_send(self, message) -> None:
        """Hand a freshly-created application message to the routing layer."""
        node = self.net.nodes[message.src]
        self.metrics.on_message_created(message)
        if node.is_operational(self.now):
            self.protocol.on_app_message(self, node, message)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self, until: float) -> None:
        self.protocol.on_start(self)
        while self._pq:
            when, _, fn, args = self._pq[0]
            if when > until:
                break
            heapq.heappop(self._pq)
            self.now = when
            fn(*args)
        self.now = until
