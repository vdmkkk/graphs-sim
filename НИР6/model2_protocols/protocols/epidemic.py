"""Epidemic / gossip routing — probabilistic flooding (DTN heritage).

Each node that hears a *new* message rebroadcasts it with probability
``gossip_p`` after a short random delay. There is no overhear suppression and
no contention shaping, so coverage is bought with raw redundancy. It is the
most robust strategy under heavy node churn / intermittent links, and the
most wasteful of airtime and energy — a deliberate opposite pole to AODV.
"""

from __future__ import annotations

from .base import RoutingProtocol
from ..network.packet import Frame, FrameType, Message


class EpidemicGossip(RoutingProtocol):
    name = "EpidemicGossip"
    proactive = False

    def __init__(self, cfg, hop_limit: int = 10, gossip_p: float = 0.8):
        super().__init__(cfg)
        self.hop_limit = hop_limit
        self.gossip_p = gossip_p

    def on_app_message(self, sim, node, message: Message) -> None:
        self.already_seen(node, (message.src, message.msg_id))
        frame = self.make_data_frame(message, ttl=self.hop_limit)
        sim.mac_send(node.id, frame)

    def on_receive(self, sim, node, frame: Frame, rssi: float, snr: float) -> None:
        if frame.ftype is not FrameType.DATA:
            return
        key = (frame.src, frame.msg_id)
        if self.already_seen(node, key):
            return
        self.deliver_if_dst(sim, node, frame)
        if frame.dst == node.id or frame.ttl <= 0:
            return
        if sim.rng.random() > self.gossip_p:
            return  # gossip: probabilistically decline to relay
        delay = self.cfg.mac.proc_delay_s + sim.rng.uniform(0.0, self.cfg.mac.backoff_max_s)
        relay = frame.copy_for_relay(sender=node.id, next_hop=-1)
        sim.schedule(delay, sim.mac_send, node.id, relay)
