"""OLSR — Optimised Link State Routing (proactive, with real MPR flooding).

Every node periodically:

  * emits **HELLO** to sense symmetric 1-hop links and announce its own MPR
    set; from neighbours' HELLOs it learns the 2-hop neighbourhood,
  * selects **MPRs** — a minimal subset of 1-hop neighbours that covers all
    2-hop neighbours (greedy set cover); only MPRs relay control traffic,
  * emits **TC** (topology control), flooded *only via MPRs*, advertising the
    nodes that selected it; from received TCs every node builds a topology
    graph and computes shortest paths.

Routes are always ready, so DATA suffers no discovery latency — but the
constant HELLO/TC traffic is heavy, and on LoRa it collides with the 1 %
duty-cycle budget. That collision is precisely the suitability question.
"""

from __future__ import annotations

import networkx as nx

from .base import RoutingProtocol
from ..network.packet import Frame, FrameType, Message


class OLSR(RoutingProtocol):
    name = "OLSR"
    proactive = True

    def __init__(self, cfg, hello_interval_s: float = 10.0,
                 tc_interval_s: float = 20.0, neighbor_timeout_s: float = 35.0,
                 route_recompute_s: float = 5.0, hop_limit: int = 16):
        super().__init__(cfg)
        self.hello_interval = hello_interval_s
        self.tc_interval = tc_interval_s
        self.neighbor_timeout = neighbor_timeout_s
        self.recompute = route_recompute_s
        self.hop_limit = hop_limit

    # -- per-node state -----------------------------------------------------
    @staticmethod
    def _st(node) -> dict:
        return node.proto.setdefault("olsr", {
            "neigh": {},          # id -> last_heard
            "sym": set(),         # symmetric 1-hop neighbours
            "two_hop": {},        # neighbour -> set(its neighbours)
            "mpr": set(),         # my selected MPRs
            "mpr_sel": set(),     # nodes that selected me as MPR
            "topo": {},           # originator -> {"adv": set, "seq", "expire"}
            "tc_seen": {},        # originator -> last seq
            "ansn": 0,            # advertised neighbour sequence number
            "routes": {},         # dst -> next_hop
        })

    # -- startup: stagger periodic control to avoid synchronised collisions -
    def on_start(self, sim) -> None:
        for node in sim.net.nodes:
            self._st(node)
            sim.schedule_timer(node.id, sim.rng.uniform(0, self.hello_interval),
                               self._hello_tick)
            sim.schedule_timer(node.id, sim.rng.uniform(0, self.tc_interval),
                               self._tc_tick)
            sim.schedule_timer(node.id, sim.rng.uniform(0, self.recompute),
                               self._recompute_tick)

    # -- HELLO --------------------------------------------------------------
    def _hello_tick(self, sim, node) -> None:
        st = self._st(node)
        self._expire_neighbours(st, sim.now)
        neighbours = sorted(st["neigh"].keys())
        frame = Frame(
            ftype=FrameType.HELLO, src=node.id, dst=-1, sender=node.id,
            next_hop=-1, payload_bytes=self.cfg.header_bytes + len(neighbours),
            ttl=1, is_control=True,
            meta={"neighbours": neighbours, "mpr": set(st["mpr"])},
        )
        sim.mac_send(node.id, frame)
        sim.schedule_timer(node.id, self.hello_interval, self._hello_tick)

    def _on_hello(self, sim, node, frame: Frame) -> None:
        st = self._st(node)
        s = frame.sender
        st["neigh"][s] = sim.now
        adv = set(frame.meta["neighbours"])
        st["two_hop"][s] = adv
        if node.id in adv:           # bidirectional => symmetric link
            st["sym"].add(s)
        if node.id in frame.meta["mpr"]:
            st["mpr_sel"].add(s)
        self._select_mprs(st)

    def _select_mprs(self, st: dict) -> None:
        """Greedy set cover: fewest 1-hop relays covering all 2-hop nodes."""
        n1 = set(st["sym"])
        # 2-hop set: neighbours of symmetric neighbours, minus self & 1-hop.
        n2 = set()
        for s in n1:
            n2 |= st["two_hop"].get(s, set())
        n2 -= n1
        mpr = set()
        uncovered = set(n2)
        # Greedily pick the neighbour covering the most still-uncovered 2-hop.
        while uncovered:
            best, best_cov = None, -1
            for s in n1:
                cov = len(st["two_hop"].get(s, set()) & uncovered)
                if cov > best_cov:
                    best, best_cov = s, cov
            if best is None or best_cov <= 0:
                break
            mpr.add(best)
            uncovered -= st["two_hop"].get(best, set())
        st["mpr"] = mpr

    # -- TC -----------------------------------------------------------------
    def _tc_tick(self, sim, node) -> None:
        st = self._st(node)
        # Standard OLSR: only nodes with MPR selectors originate TC.
        if st["mpr_sel"]:
            st["ansn"] += 1
            frame = Frame(
                ftype=FrameType.TC, src=node.id, dst=-1, sender=node.id,
                next_hop=-1,
                payload_bytes=self.cfg.header_bytes + len(st["mpr_sel"]),
                ttl=self.hop_limit, is_control=True, seq=st["ansn"],
                meta={"orig": node.id, "adv": set(st["mpr_sel"])},
            )
            sim.mac_send(node.id, frame)
        sim.schedule_timer(node.id, self.tc_interval, self._tc_tick)

    def _on_tc(self, sim, node, frame: Frame) -> None:
        st = self._st(node)
        orig = frame.meta["orig"]
        last = st["tc_seen"].get(orig, -1)
        if frame.seq <= last:
            return  # stale or duplicate
        st["tc_seen"][orig] = frame.seq
        st["topo"][orig] = {"adv": set(frame.meta["adv"]), "seq": frame.seq,
                            "expire": sim.now + 3 * self.tc_interval}
        # MPR flooding: forward only if we received it from an MPR selector.
        if frame.sender in st["mpr_sel"] and frame.ttl > 0:
            relay = frame.copy_for_relay(sender=node.id, next_hop=-1)
            sim.mac_send(node.id, relay)

    # -- routing table ------------------------------------------------------
    def _recompute_tick(self, sim, node) -> None:
        self._recompute_routes(node, sim.now)
        sim.schedule_timer(node.id, self.recompute, self._recompute_tick)

    def _recompute_routes(self, node, now: float) -> None:
        st = self._st(node)
        self._expire_neighbours(st, now)
        G = nx.Graph()
        for s in st["sym"]:
            G.add_edge(node.id, s)
        for orig, entry in list(st["topo"].items()):
            if entry["expire"] < now:
                st["topo"].pop(orig, None)
                continue
            for adv in entry["adv"]:
                G.add_edge(orig, adv)
        routes = {}
        if node.id in G:
            paths = nx.single_source_shortest_path(G, node.id)
            for dst, path in paths.items():
                if len(path) >= 2:
                    routes[dst] = path[1]
        st["routes"] = routes

    def _expire_neighbours(self, st: dict, now: float) -> None:
        dead = [n for n, t in st["neigh"].items() if now - t > self.neighbor_timeout]
        for n in dead:
            st["neigh"].pop(n, None)
            st["sym"].discard(n)
            st["two_hop"].pop(n, None)
            st["mpr_sel"].discard(n)

    # -- data / dispatch ----------------------------------------------------
    def on_app_message(self, sim, node, message: Message) -> None:
        self._send_data(sim, node, message.dst, message.msg_id, message.src,
                        self.hop_limit)

    def on_receive(self, sim, node, frame: Frame, rssi: float, snr: float) -> None:
        if frame.ftype is FrameType.HELLO:
            self._on_hello(sim, node, frame)
        elif frame.ftype is FrameType.TC:
            self._on_tc(sim, node, frame)
        elif frame.ftype is FrameType.DATA:
            if frame.next_hop != node.id:
                return
            if frame.dst == node.id:
                self.deliver_if_dst(sim, node, frame)
                return
            if self.already_seen(node, ("d", frame.src, frame.msg_id, frame.hop_count)):
                return
            if frame.ttl > 0:
                self._send_data(sim, node, frame.dst, frame.msg_id, frame.src,
                                frame.ttl, prev=frame)

    def _send_data(self, sim, node, dst, msg_id, src, ttl, prev: Frame | None = None) -> None:
        st = self._st(node)
        nh = st["routes"].get(dst)
        if nh is None:
            return  # no proactive route -> drop (counts against PDR)
        if prev is not None:
            out = prev.copy_for_relay(sender=node.id, next_hop=nh)
        else:
            out = Frame(
                ftype=FrameType.DATA, src=src, dst=dst, sender=node.id,
                next_hop=nh,
                payload_bytes=self.cfg.data_payload_bytes + self.cfg.header_bytes,
                msg_id=msg_id, ttl=ttl, is_control=False,
            )
        sim.mac_send(node.id, out)
