"""Node model: position, radio state, energy budget and duty-cycle accounting.

Nodes are *stateful* — the central difference from Model 1, where vertices
were inert. A node can run out of battery, exhaust its legal duty-cycle
airtime, be asleep, or be administratively failed by a scenario. Routing
protocols must cope with all of this.
"""

from __future__ import annotations

import enum
from collections import deque

from ..config import EnergyModel, MACConfig


class RadioState(enum.Enum):
    SLEEP = "SLEEP"
    IDLE = "IDLE"
    RX = "RX"
    TX = "TX"


class Node:
    """A single LoRa mesh node."""

    def __init__(self, node_id: int, x: float, y: float,
                 energy: EnergyModel, mac: MACConfig):
        self.id = node_id
        self.x = x
        self.y = y
        self.energy = energy
        self.mac = mac

        # --- liveness ---
        self.alive = True
        self.fail_at: float | None = None
        self.recover_at: float | None = None

        # --- radio / half-duplex bookkeeping ---
        self.state = RadioState.RX           # mesh nodes listen by default
        self.tx_start: float = -1.0
        self.tx_end: float = -1.0
        self.n_tx = 0
        self.n_rx = 0

        # --- energy ---
        self.budget_j = energy.joules(energy.battery_mah, 3600.0)  # mAh -> J
        self.consumed_tx_j = 0.0             # extra energy spent transmitting
        self.total_airtime_s = 0.0           # cumulative time-on-air
        self.battery_dead_at: float | None = None

        # --- duty cycle: sliding window of (end_time, airtime) ---
        self._duty: deque[tuple[float, float]] = deque()

        # --- routing-protocol scratch space ---
        self.proto: dict = {}

    # ------------------------------------------------------------------
    # liveness
    # ------------------------------------------------------------------
    def is_operational(self, now: float) -> bool:
        """Alive, battery not flat, and not in a scheduled-failure window."""
        if self.battery_dead_at is not None and now >= self.battery_dead_at:
            return False
        return self.alive

    # ------------------------------------------------------------------
    # half-duplex
    # ------------------------------------------------------------------
    def transmitted_during(self, start: float, end: float) -> bool:
        """Did this node's own transmission overlap [start, end]? (deaf if so)"""
        if self.tx_start < 0:
            return False
        return self.tx_start < end and self.tx_end > start

    # ------------------------------------------------------------------
    # duty cycle
    # ------------------------------------------------------------------
    def _prune_duty(self, now: float) -> None:
        window_start = now - self.mac.duty_window_s
        while self._duty and self._duty[0][0] < window_start:
            self._duty.popleft()

    def airtime_used(self, now: float) -> float:
        self._prune_duty(now)
        return sum(a for _, a in self._duty)

    def duty_allows(self, now: float, airtime: float) -> bool:
        budget = self.mac.duty_cycle * self.mac.duty_window_s
        return self.airtime_used(now) + airtime <= budget + 1e-9

    # ------------------------------------------------------------------
    # transmission accounting
    # ------------------------------------------------------------------
    def record_transmission(self, start: float, end: float, airtime: float) -> None:
        self.tx_start, self.tx_end = start, end
        self.n_tx += 1
        self.total_airtime_s += airtime
        self._duty.append((end, airtime))
        # Extra energy vs. the always-on RX baseline (which is charged globally).
        delta_ma = self.energy.current_tx_ma - self.energy.current_rx_ma
        self.consumed_tx_j += self.energy.joules(delta_ma, airtime)
        self._check_battery(end)

    def record_reception(self) -> None:
        self.n_rx += 1

    def _check_battery(self, now: float) -> None:
        if self.battery_dead_at is not None:
            return
        rx_power_w = self.energy.joules(self.energy.current_rx_ma, 1.0)
        rx_baseline = rx_power_w * now
        if rx_baseline + self.consumed_tx_j >= self.budget_j:
            self.battery_dead_at = now

    def total_energy_j(self, sim_time: float) -> float:
        """Total energy = always-on RX baseline + extra TX energy."""
        baseline = self.energy.joules(self.energy.current_rx_ma, sim_time)
        return baseline + self.consumed_tx_j

    def estimated_lifetime_s(self, sim_time: float) -> float:
        """Extrapolated battery lifetime at the observed average power draw."""
        energy = self.total_energy_j(sim_time)
        if energy <= 0:
            return float("inf")
        avg_power_w = energy / sim_time
        return self.budget_j / avg_power_w

    def __repr__(self) -> str:  # pragma: no cover
        return f"Node({self.id} @ ({self.x:.0f},{self.y:.0f}))"
