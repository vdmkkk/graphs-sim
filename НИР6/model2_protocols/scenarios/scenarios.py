"""Experiment scenarios: traffic patterns and node-state dynamics.

A scenario answers "under what conditions are we judging the protocol?".
Beyond a static baseline we stress the stack the way an offline deployment
really fails:

  * **churn**   — nodes drop out and rejoin (flat batteries, people moving,
                  obstructions). Tests resilience and route repair.
  * **partition** — a large fraction fails at once, splitting the network.
  * **load sweep** — offered traffic is increased to map the latency-vs-
                  throughput trade-off and find the congestion collapse point.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..network.deployment import DeploymentScenario
from ..network.packet import Message


@dataclass
class TrafficModel:
    """Poisson message arrivals between reachable random node pairs."""

    rate_per_min: float = 6.0     # network-wide offered load
    warmup_s: float = 60.0        # let proactive protocols converge first
    cooldown_frac: float = 0.85   # stop injecting near the end so tails drain

    def generate(self, network, rng: random.Random, sim_time: float,
                 payload_bytes: int) -> list[tuple[float, Message]]:
        pairs = network.reachable_pairs()
        if not pairs:
            return []
        last = sim_time * self.cooldown_frac
        rate_per_s = self.rate_per_min / 60.0
        out: list[tuple[float, Message]] = []
        t = self.warmup_s
        mid = 0
        while t < last:
            t += rng.expovariate(rate_per_s) if rate_per_s > 0 else float("inf")
            if t >= last:
                break
            src, dst = rng.choice(pairs)
            if rng.random() < 0.5:
                src, dst = dst, src
            out.append((t, Message(mid, src, dst, t, payload_bytes)))
            mid += 1
        return out


@dataclass
class Scenario:
    """A named experimental condition."""

    name: str
    deployment: DeploymentScenario = DeploymentScenario.UNIFORM
    num_nodes: int = 40
    traffic: TrafficModel = field(default_factory=TrafficModel)
    churn_rate_per_min: float = 0.0   # node failures injected per minute
    mean_down_s: float = 120.0        # how long a failed node stays down
    partition_fraction: float = 0.0   # fraction failed simultaneously at mid-run

    def node_dynamics(self, network, rng: random.Random, sim_time: float
                      ) -> list[tuple[float, int, bool]]:
        """Return [(time, node_id, alive)] events toggling node liveness."""
        events: list[tuple[float, int, bool]] = []
        n = network.n

        # Continuous churn: exponential failures, each recovering after a while.
        if self.churn_rate_per_min > 0:
            rate_per_s = self.churn_rate_per_min / 60.0
            t = self.traffic.warmup_s
            while t < sim_time:
                t += rng.expovariate(rate_per_s)
                if t >= sim_time:
                    break
                node_id = rng.randrange(n)
                down = rng.expovariate(1.0 / self.mean_down_s)
                events.append((t, node_id, False))
                if t + down < sim_time:
                    events.append((t + down, node_id, True))

        # One-shot partition: knock out a fraction at the midpoint.
        if self.partition_fraction > 0:
            mid = sim_time * 0.5
            victims = rng.sample(range(n), int(self.partition_fraction * n))
            for v in victims:
                events.append((mid, v, False))

        events.sort(key=lambda e: e[0])
        return events


# ---------------------------------------------------------------------------
# factory helpers
# ---------------------------------------------------------------------------

def baseline_scenario(num_nodes: int = 40,
                      deployment: DeploymentScenario = DeploymentScenario.UNIFORM
                      ) -> Scenario:
    return Scenario("baseline", deployment, num_nodes,
                    TrafficModel(rate_per_min=6.0))


def churn_scenario(num_nodes: int = 40,
                   deployment: DeploymentScenario = DeploymentScenario.UNIFORM
                   ) -> Scenario:
    return Scenario("churn", deployment, num_nodes,
                    TrafficModel(rate_per_min=6.0),
                    churn_rate_per_min=4.0, mean_down_s=120.0)


def partition_scenario(num_nodes: int = 40,
                       deployment: DeploymentScenario = DeploymentScenario.UNIFORM
                       ) -> Scenario:
    return Scenario("partition", deployment, num_nodes,
                    TrafficModel(rate_per_min=6.0),
                    partition_fraction=0.35)


def load_sweep_rates() -> list[float]:
    """Offered-load grid (messages/min) for the latency-throughput sweep."""
    return [2.0, 6.0, 12.0, 24.0, 48.0, 96.0]


SCENARIOS = {
    "baseline": baseline_scenario,
    "churn": churn_scenario,
    "partition": partition_scenario,
}
