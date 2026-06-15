"""Idealised baseline — the physical upper bound, not a real protocol.

This "protocol" cheats: it has instant, perfect, zero-cost knowledge of the
ground-truth topology and routes each message along the least-cost path
(minimising the physically-justified ETX*airtime weight) with *no* control
traffic and *no* discovery delay.

It is still bound by the real channel — collisions, half-duplex, duty cycle —
so it represents the best achievable performance on this physical network.
The gap between each real protocol and this ceiling is the headline result:
how much of the physically-possible performance does the protocol actually
capture?
"""

from __future__ import annotations

import networkx as nx

from .base import RoutingProtocol
from ..network.packet import Frame, FrameType, Message


class IdealBaseline(RoutingProtocol):
    name = "Ideal"
    proactive = False

    def __init__(self, cfg, hop_limit: int = 32):
        super().__init__(cfg)
        self.hop_limit = hop_limit
        self._cache: dict[tuple[int, int], list[int]] = {}
        self._graph_stamp = -1.0

    def on_app_message(self, sim, node, message: Message) -> None:
        nh = self._next_hop(sim, node.id, message.dst)
        if nh is None:
            return
        frame = self.make_data_frame(message, ttl=self.hop_limit, next_hop=nh)
        sim.mac_send(node.id, frame)

    def on_receive(self, sim, node, frame: Frame, rssi: float, snr: float) -> None:
        if frame.ftype is not FrameType.DATA or frame.next_hop != node.id:
            return
        if frame.dst == node.id:
            self.deliver_if_dst(sim, node, frame)
            return
        if frame.ttl <= 0:
            return
        nh = self._next_hop(sim, node.id, frame.dst)
        if nh is None:
            return
        relay = frame.copy_for_relay(sender=node.id, next_hop=nh)
        sim.mac_send(node.id, relay)

    # -- oracle routing -----------------------------------------------------
    def _next_hop(self, sim, src: int, dst: int) -> int | None:
        # Recompute the oracle graph only when the topology may have changed.
        if sim.now != self._graph_stamp:
            self._graph = sim.net.ground_truth_graph(only_operational=True)
            self._cache.clear()
            self._graph_stamp = sim.now
        key = (src, dst)
        if key not in self._cache:
            try:
                path = nx.shortest_path(self._graph, src, dst, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path = None
            self._cache[key] = path
        path = self._cache[key]
        if not path or len(path) < 2:
            return None
        return path[1]
