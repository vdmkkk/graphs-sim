"""Network-level metric collection — the quantities the NIR actually reports.

Where Model 1 measured *computational* effort (nodes visited, edges relaxed),
Model 2 measures *communication* outcomes:

  * **PDR** — packet (message) delivery ratio: the headline reliability number.
  * **Latency** — end-to-end delay of delivered messages (mean / median / p95).
  * **Goodput** — delivered application payload per second.
  * **Overhead** — control frames and airtime spent *not* carrying new payload.
  * **Redundancy** — air transmissions per delivered message.
  * **Energy** — joules per delivered bit and extrapolated node lifetime.
  * **Channel** — airtime utilisation and collision rate.
  * **Resilience** — reported by sweeping scenarios (see scenarios module).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..network.packet import Frame, FrameType, Message


@dataclass
class Metrics:
    cfg: object
    messages: dict = field(default_factory=dict)        # msg_id -> Message
    latencies: list = field(default_factory=list)
    hop_counts: list = field(default_factory=list)

    created: int = 0
    delivered: int = 0

    frames_tx: int = 0           # total over-the-air transmissions
    data_frames_tx: int = 0      # DATA frames (incl. relays/duplicates)
    control_frames_tx: int = 0   # routing-overhead frames
    total_airtime_s: float = 0.0
    control_airtime_s: float = 0.0

    receptions: int = 0          # successful decodes across all nodes
    suppressed: int = 0          # managed-flood rebroadcasts cancelled
    drops: dict = field(default_factory=dict)  # reason -> count

    # ------------------------------------------------------------------
    # event hooks (called by the engine / protocols)
    # ------------------------------------------------------------------
    def on_message_created(self, message: Message) -> None:
        self.messages[message.msg_id] = message
        self.created += 1

    def on_delivered(self, msg_id: int, now: float, hops: int) -> None:
        msg = self.messages.get(msg_id)
        if msg is None or msg.delivered:
            return  # count each message's first delivery only
        msg.delivered = True
        msg.delivered_at = now
        msg.hops = hops
        self.delivered += 1
        self.latencies.append(now - msg.created_at)
        self.hop_counts.append(hops)

    def on_air(self, frame: Frame, airtime: float, now: float) -> None:
        self.frames_tx += 1
        self.total_airtime_s += airtime
        if frame.is_control:
            self.control_frames_tx += 1
            self.control_airtime_s += airtime
        else:
            self.data_frames_tx += 1

    def on_receive(self, frame: Frame, receiver: int, now: float) -> None:
        self.receptions += 1

    def on_suppressed(self) -> None:
        self.suppressed += 1

    def on_mac_drop(self, frame: Frame, reason: str) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    @staticmethod
    def _pct(values: list, q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        k = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
        return ordered[k]

    def summarize(self, network, channel, sim_time: float) -> dict:
        cfg = self.cfg
        pdr = self.delivered / self.created if self.created else 0.0

        lat_mean = statistics.fmean(self.latencies) if self.latencies else float("nan")
        lat_med = statistics.median(self.latencies) if self.latencies else float("nan")
        lat_p95 = self._pct(self.latencies, 0.95)

        payload_bits = self.delivered * cfg.data_payload_bytes * 8
        goodput_bps = payload_bits / sim_time if sim_time else 0.0

        total_energy = sum(n.total_energy_j(sim_time) for n in network.nodes)
        energy_per_bit = (total_energy / payload_bits) if payload_bits else float("inf")
        lifetimes = [n.estimated_lifetime_s(sim_time) for n in network.nodes]
        first_death = min((n.battery_dead_at for n in network.nodes
                           if n.battery_dead_at is not None), default=None)

        ch_util = self.total_airtime_s / sim_time if sim_time else 0.0
        coll_rate = (channel.collided_transmissions / channel.total_transmissions
                     if channel.total_transmissions else 0.0)
        coll_per_tx = (channel.total_collisions / channel.total_transmissions
                       if channel.total_transmissions else 0.0)

        return {
            # reliability
            "messages_created": self.created,
            "messages_delivered": self.delivered,
            "pdr": round(pdr, 4),
            # latency
            "latency_mean_s": round(lat_mean, 3) if self.latencies else None,
            "latency_median_s": round(lat_med, 3) if self.latencies else None,
            "latency_p95_s": round(lat_p95, 3) if self.latencies else None,
            "mean_hops": round(statistics.fmean(self.hop_counts), 2) if self.hop_counts else None,
            # throughput
            "goodput_bps": round(goodput_bps, 3),
            # overhead
            "frames_tx": self.frames_tx,
            "data_frames_tx": self.data_frames_tx,
            "control_frames_tx": self.control_frames_tx,
            "control_airtime_frac": round(self.control_airtime_s / self.total_airtime_s, 4)
            if self.total_airtime_s else 0.0,
            "tx_per_delivery": round(self.frames_tx / self.delivered, 2) if self.delivered else None,
            "ctrl_frames_per_delivery": round(self.control_frames_tx / self.delivered, 2)
            if self.delivered else None,
            "suppressed_rebroadcasts": self.suppressed,
            # channel
            "channel_utilisation": round(ch_util, 4),
            "collision_rate": round(coll_rate, 4),
            "collisions_per_tx": round(coll_per_tx, 4),
            "drops_duty_cycle": self.drops.get("duty_cycle", 0),
            "drops_channel_busy": self.drops.get("channel_busy", 0),
            # energy
            "total_energy_j": round(total_energy, 1),
            "energy_per_bit_mj": round(energy_per_bit * 1000, 4) if payload_bits else None,
            "mean_node_lifetime_h": round(statistics.fmean(lifetimes) / 3600, 2),
            "first_node_death_s": round(first_death, 1) if first_death is not None else None,
        }
