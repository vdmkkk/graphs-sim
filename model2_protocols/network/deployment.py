"""Geographic node deployment and the connectivity it induces.

LoRa is range-limited, so *where* nodes sit determines who can hear whom —
unlike Model 1's abstract topologies. We place nodes on a 2-D plane (metres),
derive each potential link from the PHY (path loss -> SNR -> reliability),
and expose:

  * ``feasible_receivers`` / ``sample_link`` for the channel,
  * a *ground-truth* weighted graph for the idealised baseline and for
    sanity-checking reachability,

with a link *cost* that is physically justified:

    cost(u->v) = airtime(v) / P_reception(u->v)        [ "expected airtime
                                                          per delivery" ]

i.e. the classic ETX metric scaled by time-on-air. A link is cheap when it
is both reliable (high delivery probability) and fast (low airtime). This is
the weight a real LoRa routing protocol would want to minimise.
"""

from __future__ import annotations

import enum
import math
import random

import networkx as nx

from ..config import SimConfig
from ..radio.lora_phy import (
    LoRaLink,
    path_loss_db,
    rssi_dbm,
    reception_probability,
)
from .node import Node


class DeploymentScenario(enum.Enum):
    UNIFORM = "uniform"        # nodes scattered uniformly (open area)
    CLUSTERED = "clustered"    # villages / camps with sparse inter-links
    CORRIDOR = "corridor"      # along a road / valley (near-linear mesh)
    GRID = "grid"              # planned regular placement


class Network:
    """Holds nodes, geometry and the feasibility graph they induce."""

    def __init__(self, nodes: list[Node], cfg: SimConfig,
                 routing_pr_threshold: float = 0.7):
        self.nodes = nodes
        self.cfg = cfg
        self.n = len(nodes)
        self.routing_pr_threshold = routing_pr_threshold
        self._dist: dict[tuple[int, int], float] = {}
        # Two notions of a link, on purpose:
        #   _audible  — node can physically hear the sender (>= sensitivity);
        #               used by the channel for reception AND interference, so
        #               opportunistic long shots (which flooding exploits) and
        #               collisions are both modelled.
        #   _routing  — link reliable enough to build a route over (mean
        #               delivery prob >= threshold); used by routing tables and
        #               the idealised oracle. Real link-quality routing does
        #               exactly this: it does not route over 50/50 links.
        self._audible: dict[int, list[int]] = {i: [] for i in range(self.n)}
        self._routing: dict[int, list[int]] = {i: [] for i in range(self.n)}
        self._mean_pr: dict[tuple[int, int], float] = {}
        self._build_links()

    # ------------------------------------------------------------------
    def distance(self, i: int, j: int) -> float:
        key = (i, j) if i < j else (j, i)
        d = self._dist.get(key)
        if d is None:
            a, b = self.nodes[i], self.nodes[j]
            d = math.hypot(a.x - b.x, a.y - b.y)
            self._dist[key] = d
        return d

    def _build_links(self) -> None:
        phy, prop = self.cfg.phy, self.cfg.propagation
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                d = self.distance(i, j)
                link = LoRaLink(d, phy, prop)
                if link.is_feasible():
                    pr = link.mean_reception_probability()
                    self._audible[i].append(j)
                    self._mean_pr[(i, j)] = pr
                    if pr >= self.routing_pr_threshold:
                        self._routing[i].append(j)

    # ------------------------------------------------------------------
    # channel collaborator API
    # ------------------------------------------------------------------
    def feasible_receivers(self, sender: int) -> list[int]:
        """Operational nodes that can, on average, physically hear *sender*."""
        out = []
        for j in self._audible[sender]:
            node = self.nodes[j]
            if node.alive and node.battery_dead_at is None:
                out.append(j)
        return out

    def sample_link(self, sender: int, receiver: int, rng: random.Random) -> tuple[float, float]:
        """Sample instantaneous (rssi, snr) including log-normal shadowing."""
        phy, prop = self.cfg.phy, self.cfg.propagation
        d = self.distance(sender, receiver)
        shadow = rng.gauss(0.0, prop.shadowing_sigma_db)
        rssi = rssi_dbm(d, phy, prop, shadow)
        snr = rssi - _noise_floor(phy)
        return rssi, snr

    # ------------------------------------------------------------------
    # routing helpers
    # ------------------------------------------------------------------
    def link_cost(self, u: int, v: int) -> float | None:
        """Physically-justified ETX-style cost (expected airtime / delivery)."""
        pr = self._mean_pr.get((u, v))
        if pr is None or pr <= 0:
            return None
        frame_bytes = self.cfg.data_payload_bytes + self.cfg.header_bytes
        link = LoRaLink(self.distance(u, v), self.cfg.phy, self.cfg.propagation)
        return link.airtime(frame_bytes) / pr

    def ground_truth_graph(self, only_operational: bool = True) -> nx.Graph:
        """Weighted undirected graph of *routing-quality* links (oracle view)."""
        G = nx.Graph()
        for i in range(self.n):
            if only_operational and not self.nodes[i].alive:
                continue
            G.add_node(i)
        for i in range(self.n):
            if only_operational and not self.nodes[i].alive:
                continue
            for j in self._routing[i]:
                if j <= i:
                    continue
                if only_operational and not self.nodes[j].alive:
                    continue
                c_ij = self.link_cost(i, j) or math.inf
                c_ji = self.link_cost(j, i) or math.inf
                G.add_edge(i, j, weight=min(c_ij, c_ji))
        return G

    def mean_degree(self) -> float:
        """Mean routing-quality degree (the connectivity routing can rely on)."""
        total = sum(len(v) for v in self._routing.values())
        return total / self.n if self.n else 0.0

    def mean_audible_degree(self) -> float:
        total = sum(len(v) for v in self._audible.values())
        return total / self.n if self.n else 0.0

    def reachable_pairs(self) -> list[tuple[int, int]]:
        """All (s,t) connected via routing-quality links (a fair PDR target)."""
        G = self.ground_truth_graph(only_operational=False)
        pairs = []
        for comp in nx.connected_components(G):
            comp = list(comp)
            for a in range(len(comp)):
                for b in range(a + 1, len(comp)):
                    pairs.append((comp[a], comp[b]))
        return pairs


def _noise_floor(phy) -> float:
    from ..radio.lora_phy import noise_floor_dbm
    return noise_floor_dbm(phy)


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------

def reliable_range_m(cfg: SimConfig, pr_threshold: float = 0.5) -> float:
    """Largest distance at which the *mean* link still decodes with prob >= thr."""
    phy, prop = cfg.phy, cfg.propagation
    lo, hi = 1.0, 50_000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        rssi = rssi_dbm(mid, phy, prop)
        snr = rssi - _noise_floor(phy)
        if reception_probability(snr, rssi, phy) >= pr_threshold:
            lo = mid
        else:
            hi = mid
    return lo


def build_network(
    cfg: SimConfig,
    num_nodes: int,
    scenario: DeploymentScenario = DeploymentScenario.UNIFORM,
    target_degree: float = 7.0,
    routing_pr_threshold: float = 0.7,
    rng: random.Random | None = None,
) -> Network:
    """Place ``num_nodes`` so the mean *routing* degree is ~``target_degree``.

    The area is sized from the PHY-derived reliable range so the network is a
    genuine *multi-hop* mesh (not a single broadcast domain, not disconnected).
    """
    rng = rng or random.Random(cfg.seed)
    # Size the area against the routing-quality range so the routing graph
    # (not just the audible graph) reaches the target connectivity.
    r = reliable_range_m(cfg, pr_threshold=routing_pr_threshold)
    # Uniform density: E[neighbours] = (N/Area) * pi * r^2  => solve for side.
    area = num_nodes * math.pi * r * r / max(target_degree, 1.0)
    side = math.sqrt(area)

    positions: list[tuple[float, float]] = []

    if scenario is DeploymentScenario.UNIFORM:
        for _ in range(num_nodes):
            positions.append((rng.uniform(0, side), rng.uniform(0, side)))

    elif scenario is DeploymentScenario.GRID:
        cols = math.ceil(math.sqrt(num_nodes))
        step = side / max(cols - 1, 1)
        jitter = step * 0.15
        for k in range(num_nodes):
            gx, gy = k % cols, k // cols
            positions.append((gx * step + rng.uniform(-jitter, jitter),
                              gy * step + rng.uniform(-jitter, jitter)))

    elif scenario is DeploymentScenario.CORRIDOR:
        # Long, thin deployment: width ~ one reliable hop -> near-linear mesh.
        length = side * 2.5
        width = r * 0.8
        for _ in range(num_nodes):
            positions.append((rng.uniform(0, length), rng.uniform(0, width)))

    elif scenario is DeploymentScenario.CLUSTERED:
        n_clusters = max(2, round(num_nodes / 12))
        centres = [(rng.uniform(0, side * 1.4), rng.uniform(0, side * 1.4))
                   for _ in range(n_clusters)]
        spread = r * 0.45
        for k in range(num_nodes):
            cx, cy = centres[k % n_clusters]
            positions.append((cx + rng.gauss(0, spread), cy + rng.gauss(0, spread)))

    nodes = [Node(i, x, y, cfg.energy, cfg.mac) for i, (x, y) in enumerate(positions)]
    return Network(nodes, cfg, routing_pr_threshold=routing_pr_threshold)
