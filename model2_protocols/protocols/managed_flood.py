"""Managed flooding — the strategy Meshtastic actually ships.

Every node rebroadcasts each message at most once, but two mechanisms tame
the broadcast storm:

  * **SNR-based contention window** — a node that received the frame *weakly*
    (low SNR, i.e. it is far from the sender) waits a *shorter* time before
    rebroadcasting, so the relay that extends coverage furthest goes first.
  * **Overhear suppression** — if, while waiting, a node hears someone else
    rebroadcast the same message, it cancels its own rebroadcast.

No routing tables, no discovery: robust and zero-control, at the price of
redundant transmissions. The natural baseline for LoRa mesh.
"""

from __future__ import annotations

from .base import RoutingProtocol
from ..network.packet import Frame, FrameType, Message


class ManagedFlood(RoutingProtocol):
    name = "ManagedFlood"
    proactive = False

    def __init__(self, cfg, hop_limit: int = 8, cw_max_s: float = 2.0):
        super().__init__(cfg)
        self.hop_limit = hop_limit
        self.cw_max_s = cw_max_s

    def on_app_message(self, sim, node, message: Message) -> None:
        self.already_seen(node, (message.src, message.msg_id))  # don't relay our own
        frame = self.make_data_frame(message, ttl=self.hop_limit)
        sim.mac_send(node.id, frame)

    def on_receive(self, sim, node, frame: Frame, rssi: float, snr: float) -> None:
        if frame.ftype is not FrameType.DATA:
            return
        key = (frame.src, frame.msg_id)
        pending = node.proto.setdefault("mf_pending", {})

        if (key in self._seen_key(node)):
            # Duplicate: suppress our own scheduled rebroadcast, if any.
            if key in pending:
                pending[key]["cancelled"] = True
                pending.pop(key, None)
                sim.metrics.on_suppressed()
            return

        self.already_seen(node, key)

        if self.deliver_if_dst(sim, node, frame):
            return
        if frame.ttl <= 0:
            return

        token = {"cancelled": False, "frame": frame}
        pending[key] = token
        delay = self._contention_delay(sim, snr)
        sim.schedule_timer(node.id, delay, self._rebroadcast, key)

    def _contention_delay(self, sim, snr: float) -> float:
        """Lower SNR -> shorter wait (favour range-extending relays)."""
        margin = snr - self.cfg.phy.required_snr()
        # Normalise margin into [0, 1]; 0 dB margin -> go first, >=20 dB -> last.
        snr_factor = max(0.0, min(1.0, margin / 20.0))
        jitter = sim.rng.uniform(0.0, self.cfg.mac.backoff_max_s)
        return self.cfg.mac.proc_delay_s + snr_factor * self.cw_max_s + jitter

    def _rebroadcast(self, sim, node, key) -> None:
        pending = node.proto.get("mf_pending", {})
        token = pending.pop(key, None)
        if token is None or token["cancelled"]:
            return
        if not node.is_operational(sim.now):
            return
        relay = token["frame"].copy_for_relay(sender=node.id, next_hop=-1)
        sim.mac_send(node.id, relay)
