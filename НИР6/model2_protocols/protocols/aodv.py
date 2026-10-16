"""AODV — Ad-hoc On-demand Distance Vector routing (reactive).

Routes are discovered only when needed:

  * a source with no route floods a **RREQ**; every relay records a reverse
    route back toward the source and rebroadcasts it (with dedup),
  * the destination (or any node holding a fresh route) answers with a
    **RREP** that is unicast back along the reverse path, installing forward
    routes hop-by-hop,
  * **DATA** is then unicast hop-by-hop along the established route.

Low steady-state overhead (nothing is sent when idle) but every new flow pays
a discovery latency, and route breaks under churn force re-discovery — exactly
the tension we want to measure against the proactive and flooding strategies.
"""

from __future__ import annotations

from .base import RoutingProtocol
from ..network.packet import Frame, FrameType, Message


class AODV(RoutingProtocol):
    name = "AODV"
    proactive = False

    def __init__(self, cfg, hop_limit: int = 12, route_lifetime_s: float = 60.0,
                 rreq_timeout_s: float = 2.5, max_retries: int = 2):
        super().__init__(cfg)
        self.hop_limit = hop_limit
        self.route_lifetime = route_lifetime_s
        self.rreq_timeout = rreq_timeout_s
        self.max_retries = max_retries
        self.ctrl_bytes = cfg.header_bytes

    # -- per-node state -----------------------------------------------------
    @staticmethod
    def _st(node) -> dict:
        return node.proto.setdefault("aodv", {
            "routes": {},        # dst -> {next_hop, hops, seq, expire}
            "rreq_seen": set(),  # (orig, rreq_id)
            "rreq_id": 0,
            "seq": 0,
            "buffer": {},        # dst -> [Message, ...]
            "retries": {},       # dst -> count
        })

    def _valid_route(self, node, dst, now):
        r = self._st(node)["routes"].get(dst)
        if r and r["expire"] > now:
            return r
        return None

    def _install_route(self, node, dst, next_hop, hops, seq, now):
        routes = self._st(node)["routes"]
        cur = routes.get(dst)
        # Prefer fresher sequence number, then fewer hops (standard AODV rule).
        if cur and (cur["seq"] > seq or (cur["seq"] == seq and cur["hops"] <= hops)):
            cur["expire"] = now + self.route_lifetime
            return
        routes[dst] = {"next_hop": next_hop, "hops": hops, "seq": seq,
                       "expire": now + self.route_lifetime}

    # -- application --------------------------------------------------------
    def on_app_message(self, sim, node, message: Message) -> None:
        route = self._valid_route(node, message.dst, sim.now)
        if route:
            self._forward_data(sim, node, message.dst, message.msg_id,
                               message.src, self.hop_limit)
        else:
            self._st(node)["buffer"].setdefault(message.dst, []).append(message)
            self._start_rreq(sim, node, message.dst)

    # -- route discovery ----------------------------------------------------
    def _start_rreq(self, sim, node, dst, attempt: int = 0) -> None:
        st = self._st(node)
        if self._valid_route(node, dst, sim.now):
            self._flush_buffer(sim, node, dst)
            return
        st["rreq_id"] += 1
        st["seq"] += 1
        rid = st["rreq_id"]
        st["rreq_seen"].add((node.id, rid))
        frame = Frame(
            ftype=FrameType.RREQ, src=node.id, dst=dst, sender=node.id,
            next_hop=-1, payload_bytes=self.ctrl_bytes, ttl=self.hop_limit,
            is_control=True, seq=st["seq"],
            meta={"orig": node.id, "rreq_id": rid, "hop_metric": 0,
                  "orig_seq": st["seq"]},
        )
        sim.mac_send(node.id, frame)
        sim.schedule_timer(node.id, self.rreq_timeout, self._rreq_timeout, dst, attempt)

    def _rreq_timeout(self, sim, node, dst, attempt) -> None:
        if self._valid_route(node, dst, sim.now):
            self._flush_buffer(sim, node, dst)
            return
        if attempt < self.max_retries:
            self._start_rreq(sim, node, dst, attempt + 1)
        else:
            # Discovery failed: drop everything buffered for this destination.
            st = self._st(node)
            st["buffer"].pop(dst, None)

    # -- frame handling -----------------------------------------------------
    def on_receive(self, sim, node, frame: Frame, rssi: float, snr: float) -> None:
        if frame.ftype is FrameType.RREQ:
            self._on_rreq(sim, node, frame)
        elif frame.ftype is FrameType.RREP:
            self._on_rrep(sim, node, frame)
        elif frame.ftype is FrameType.DATA:
            self._on_data(sim, node, frame)

    def _on_rreq(self, sim, node, frame: Frame) -> None:
        st = self._st(node)
        orig = frame.meta["orig"]
        key = (orig, frame.meta["rreq_id"])
        # Always (re)install the reverse route, even on a duplicate seen frame,
        # then suppress re-flooding of duplicates.
        hops_to_orig = frame.meta["hop_metric"] + 1
        self._install_route(node, orig, frame.sender, hops_to_orig,
                            frame.meta["orig_seq"], sim.now)
        if key in st["rreq_seen"]:
            return
        st["rreq_seen"].add(key)

        # We can answer if we are the destination or hold a fresh route to it.
        if node.id == frame.dst:
            self._send_rrep(sim, node, orig, frame.dst, 0, st["seq"])
            return
        route = self._valid_route(node, frame.dst, sim.now)
        if route:
            self._send_rrep(sim, node, orig, frame.dst, route["hops"], route["seq"])
            return
        if frame.ttl <= 0:
            return
        relay = frame.copy_for_relay(sender=node.id, next_hop=-1)
        relay.meta["hop_metric"] = frame.meta["hop_metric"] + 1
        sim.mac_send(node.id, relay)

    def _send_rrep(self, sim, node, orig, data_dst, hops_to_dst, dst_seq) -> None:
        st = self._st(node)
        rev = self._valid_route(node, orig, sim.now)
        if not rev and node.id != orig:
            return
        next_hop = rev["next_hop"] if rev else node.id
        rrep = Frame(
            ftype=FrameType.RREP, src=data_dst, dst=orig, sender=node.id,
            next_hop=next_hop, payload_bytes=self.ctrl_bytes, ttl=self.hop_limit,
            is_control=True, seq=dst_seq,
            meta={"orig": orig, "data_dst": data_dst, "hop_metric": hops_to_dst},
        )
        sim.mac_send(node.id, rrep)

    def _on_rrep(self, sim, node, frame: Frame) -> None:
        if frame.next_hop != node.id and frame.dst != node.id:
            return  # not addressed to us
        data_dst = frame.meta["data_dst"]
        hops_to_dst = frame.meta["hop_metric"] + 1
        self._install_route(node, data_dst, frame.sender, hops_to_dst,
                            frame.seq, sim.now)

        if node.id == frame.dst:  # we are the original requester
            self._flush_buffer(sim, node, data_dst)
            return
        rev = self._valid_route(node, frame.dst, sim.now)
        if not rev:
            return
        fwd = Frame(
            ftype=FrameType.RREP, src=data_dst, dst=frame.dst, sender=node.id,
            next_hop=rev["next_hop"], payload_bytes=self.ctrl_bytes,
            ttl=frame.ttl - 1, is_control=True, seq=frame.seq,
            meta={"orig": frame.dst, "data_dst": data_dst,
                  "hop_metric": hops_to_dst},
        )
        if fwd.ttl > 0:
            sim.mac_send(node.id, fwd)

    def _on_data(self, sim, node, frame: Frame) -> None:
        if frame.next_hop != node.id:
            return
        if frame.dst == node.id:
            self.deliver_if_dst(sim, node, frame)
            return
        if self.already_seen(node, ("data", frame.src, frame.msg_id, frame.hop_count)):
            return
        if frame.ttl <= 0:
            return
        self._forward_data(sim, node, frame.dst, frame.msg_id, frame.src,
                           frame.ttl, prev=frame)

    def _forward_data(self, sim, node, dst, msg_id, src, ttl, prev: Frame | None = None) -> None:
        route = self._valid_route(node, dst, sim.now)
        if not route:
            # Route lost mid-flight: re-discover (frame is dropped meanwhile).
            self._start_rreq(sim, node, dst)
            return
        if prev is not None:
            out = prev.copy_for_relay(sender=node.id, next_hop=route["next_hop"])
        else:
            out = Frame(
                ftype=FrameType.DATA, src=src, dst=dst, sender=node.id,
                next_hop=route["next_hop"],
                payload_bytes=self.cfg.data_payload_bytes + self.cfg.header_bytes,
                msg_id=msg_id, ttl=ttl, is_control=False,
            )
        sim.mac_send(node.id, out)

    def _flush_buffer(self, sim, node, dst) -> None:
        st = self._st(node)
        msgs = st["buffer"].pop(dst, [])
        for m in msgs:
            self._forward_data(sim, node, dst, m.msg_id, m.src, self.hop_limit)
